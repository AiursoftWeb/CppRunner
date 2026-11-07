using System.Collections.Concurrent;
using System.Security.Claims;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Lang;
using Aiursoft.CppRunner.Services;
using Aiursoft.CSTools.Services;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Localization;
using Microsoft.Extensions.Logging.Abstractions;

namespace Aiursoft.CppRunner.Tests.IntegrationTests;

[TestClass]
public class ExecutionPolicyTests : TestBase
{
    [TestMethod]
    public async Task SeparateRequestsShareTwoSlotsAndQueueInOrder()
    {
        using var queue = new ExecutionQueue();
        using var commands = new Commands();
        var firstRunner = CreateRunner(queue, commands);
        var otherRunner = CreateRunner(queue, commands);
        var first = firstRunner.RunCode("one", new BashLang());
        await commands.OneStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var second = otherRunner.RunCode("two", new BashLang());
        await commands.TwoStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var third = firstRunner.RunCode("three", new BashLang());
        var fourth = otherRunner.RunCode("four", new BashLang());
        Assert.AreEqual(2, commands.Started.Count);
        Assert.IsFalse(third.IsCompleted);
        commands.Started[0].TrySetResult();
        await first.WaitAsync(TimeSpan.FromSeconds(10));
        await commands.ThreeStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        CollectionAssert.AreEqual(new[] { "one", "two", "three" }, commands.Code.ToArray());
        Assert.IsFalse(fourth.IsCompleted);
        commands.Started[1].TrySetResult();
        await second.WaitAsync(TimeSpan.FromSeconds(10));
        await commands.FourStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        commands.Started[2].TrySetResult();
        commands.Started[3].TrySetResult();
        await Task.WhenAll(third, fourth).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.AreEqual(2, commands.Peak);
    }

    [TestMethod]
    public async Task DisconnectedQueuedRequestNeverStartsAndDoesNotBlockNextRequest()
    {
        using var queue = new ExecutionQueue();
        using var commands = new Commands();
        var context = new DefaultHttpContext();
        using var disconnected = new CancellationTokenSource();
        context.RequestAborted = disconnected.Token;
        var runner = CreateRunner(queue, commands);
        var first = runner.RunCode("one", new BashLang());
        await commands.OneStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var second = runner.RunCode("two", new BashLang());
        await commands.TwoStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var abandoned = CreateRunner(queue, commands, context).RunCode("abandoned", new BashLang());
        disconnected.Cancel();
        await Assert.ThrowsAsync<OperationCanceledException>(() => abandoned);
        var next = runner.RunCode("next", new BashLang());
        commands.Started[0].TrySetResult();
        await first.WaitAsync(TimeSpan.FromSeconds(10));
        await commands.ThreeStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        CollectionAssert.AreEqual(new[] { "one", "two", "next" }, commands.Code.ToArray());
        commands.Started[1].TrySetResult();
        commands.Started[2].TrySetResult();
        await Task.WhenAll(second, next).WaitAsync(TimeSpan.FromSeconds(10));
    }

    [TestMethod]
    public async Task FailedExecutionReleasesItsSlot()
    {
        using var queue = new ExecutionQueue();
        using var commands = new Commands { FailFirst = true };
        var runner = CreateRunner(queue, commands);
        var first = runner.RunCode("one", new BashLang());
        await commands.OneStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var second = runner.RunCode("two", new BashLang());
        await commands.TwoStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var third = runner.RunCode("three", new BashLang());
        commands.Started[0].TrySetResult();
        await Assert.ThrowsExactlyAsync<InvalidOperationException>(() => first);
        await commands.ThreeStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        commands.Started[1].TrySetResult();
        commands.Started[2].TrySetResult();
        await Task.WhenAll(second, third).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.AreEqual(2, commands.Peak);
    }

    [TestMethod]
    public async Task AnonymousGpuIsRejectedBeforeQueuingOrInspectingImages()
    {
        using var queue = new ExecutionQueue();
        using var one = await queue.AcquireAsync(CancellationToken.None);
        using var two = await queue.AcquireAsync(CancellationToken.None);
        using var commands = new Commands();
        var images = new RuntimeImageWarmupTests.ImageClient { Available = true };
        var runner = CreateRunner(queue, commands, images: images);
        foreach (ILang lang in new ILang[] { new CudaLang(), new PythonWithPytorch() })
        {
            var result = await runner.RunCode("not executed", lang).WaitAsync(TimeSpan.FromSeconds(10));
            Assert.AreEqual(401, result.ResultCode);
            StringAssert.Contains(result.Error, "Sign in");
        }
        Assert.AreEqual(0, images.Inspections);
        Assert.IsEmpty(commands.Started);
    }

    [TestMethod]
    public async Task SignedInConsumerCanRunGpuCode()
    {
        using var queue = new ExecutionQueue();
        using var commands = new Commands();
        using var cache = new MemoryCache(new MemoryCacheOptions());
        var gpu = new HasGpuService(commands,
            new CacheService(cache, NullLogger<CacheService>.Instance), NullLogger<HasGpuService>.Instance);
        var context = new DefaultHttpContext
        {
            User = new ClaimsPrincipal(new ClaimsIdentity([new Claim(ClaimTypes.NameIdentifier, "consumer")], "test"))
        };
        var task = CreateRunner(queue, commands, context, gpu: gpu).RunCode("gpu", new CudaLang());
        await commands.OneStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        StringAssert.Contains(commands.RunArguments.Single(), "--gpus all");
        commands.Started[0].TrySetResult();
        Assert.AreEqual(0, (await task.WaitAsync(TimeSpan.FromSeconds(10))).ResultCode);
    }

    private RunCodeService CreateRunner(ExecutionQueue queue, Commands commands,
        HttpContext? context = null, RuntimeImageWarmupTests.ImageClient? images = null, HasGpuService? gpu = null)
    {
        var services = Server!.Services;
        return new RunCodeService(services.GetRequiredService<IConfiguration>(),
            gpu ?? services.GetRequiredService<HasGpuService>(), NullLogger<RunCodeService>.Instance,
            commands, services.GetRequiredService<CanonQueue>(),
            images ?? new RuntimeImageWarmupTests.ImageClient { Available = true },
            services.GetRequiredService<IHostApplicationLifetime>(),
            services.GetRequiredService<IStringLocalizer<RunCodeService>>(), queue,
            new ContextAccessor(context ?? new DefaultHttpContext()));
    }

    private sealed class ContextAccessor(HttpContext context) : IHttpContextAccessor
    {
        public HttpContext? HttpContext { get; set; } = context;
    }

    private sealed class Commands : CommandService, IDisposable
    {
        public List<TaskCompletionSource> Started { get; } = [];
        public ConcurrentQueue<string> Code { get; } = new();
        public List<string> RunArguments { get; } = [];
        public TaskCompletionSource OneStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource TwoStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ThreeStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource FourStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool FailFirst { get; init; }
        private int _active;
        public int Peak { get; private set; }

        public override async Task<(int code, string output, string error)> RunCommandAsync(string bin, string arg,
            string path, TimeSpan? timeout = null, bool killTimeoutProcess = true,
            IDictionary<string, string?>? environmentVariables = null)
        {
            if (!arg.StartsWith("run ", StringComparison.Ordinal))
                return (0, "NVIDIA NVIDIA-SMI GPU-123 nvidia", string.Empty);
            var start = arg.IndexOf(" -v ", StringComparison.Ordinal) + 4;
            var end = arg.IndexOf(":/app", start, StringComparison.Ordinal);
            var folder = arg[start..end];
            Code.Enqueue(await File.ReadAllTextAsync(Directory.GetFiles(folder).Single(file => Path.GetFileName(file) is "main.sh" or "main.cu")));
            var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var active = Interlocked.Increment(ref _active);
            int count;
            lock (Started)
            {
                Peak = Math.Max(Peak, active);
                Started.Add(release);
                RunArguments.Add(arg);
                count = Started.Count;
            }
            if (count == 1) OneStarted.TrySetResult();
            if (count == 2) TwoStarted.TrySetResult();
            if (count == 3) ThreeStarted.TrySetResult();
            if (count == 4) FourStarted.TrySetResult();
            try
            {
                await release.Task;
                if (FailFirst && count == 1) throw new InvalidOperationException("Execution failed");
                return (0, "done", string.Empty);
            }
            finally { Interlocked.Decrement(ref _active); }
        }

        public void Dispose()
        {
            foreach (var release in Started) release.TrySetResult();
        }
    }
}
