using System.Collections.Concurrent;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Configuration;
using Aiursoft.CppRunner.Lang;
using Aiursoft.CppRunner.Services;
using Aiursoft.CppRunner.Services.BackgroundJobs;
using Aiursoft.CSTools.Services;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace Aiursoft.CppRunner.Tests;

[TestClass]
public class RuntimeImageWarmupTests
{
    [TestMethod]
    public async Task CachedImagesDoNotRequireRegistryAuthentication()
    {
        var images = new ImageClient();
        images.Cached.Add("gcc:15.2.0");
        using var lifetime = new Lifetime();
        await Create(images, lifetime, new DockerImageSettings { RequireAuthentication = true }, new CLang()).ExecuteAsync();
        Assert.IsEmpty(images.Pulled);
        Assert.AreEqual(0, images.Inspections);
    }

    [TestMethod]
    public async Task CanonicalAliasesAreInspectedWithoutPulling()
    {
        var images = new ImageClient { Available = true };
        using var lifetime = new Lifetime();
        await Create(images, lifetime, new DockerImageSettings { RequireAuthentication = true }, new CLang()).ExecuteAsync();
        Assert.IsEmpty(images.Pulled);
        Assert.AreEqual(1, images.Inspections);
    }

    [TestMethod]
    public async Task TagsAreMatchedExactlyAndSharedImagesArePreparedOnce()
    {
        var images = new ImageClient();
        images.Cached.Add("gcc:15.2.0-extra");
        using var lifetime = new Lifetime();
        await Create(images, lifetime, new DockerImageSettings(), new CLang(), new CLang()).ExecuteAsync();
        CollectionAssert.AreEqual(new[] { "gcc:15.2.0" }, images.Pulled.ToArray());
    }

    [TestMethod]
    public async Task FailedImageDoesNotPreventOtherLanguagesAndManualRetrySkipsSuccesses()
    {
        var images = new ImageClient { FailImage = "bad:1" };
        using var lifetime = new Lifetime();
        var job = Create(images, lifetime, new DockerImageSettings(), new Language("bad:1"), new Language("good:1"));
        await Assert.ThrowsExactlyAsync<InvalidOperationException>(() => job.ExecuteAsync());
        CollectionAssert.AreEquivalent(new[] { "bad:1", "good:1" }, images.Pulled.ToArray());
        images.FailImage = null;
        await job.ExecuteAsync();
        Assert.AreEqual(1, images.Pulled.Count(image => image == "good:1"));
        Assert.AreEqual(2, images.Pulled.Count(image => image == "bad:1"));
    }

    [TestMethod]
    public async Task SlowPullsAreLimitedToTwoAndShutdownCancelsThem()
    {
        var images = new ImageClient { Block = true };
        using var lifetime = new Lifetime();
        var task = Create(images, lifetime, new DockerImageSettings(),
            new Language("one:1"), new Language("two:1"), new Language("three:1")).ExecuteAsync();
        await images.TwoStarted.Task.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.AreEqual(2, images.Pulled.Count);
        lifetime.StopApplication();
        await Assert.ThrowsAsync<OperationCanceledException>(() => task);
        Assert.AreEqual(2, images.Pulled.Count);
    }

    [TestMethod]
    public async Task AlreadyStoppingDoesNotStartPreparation()
    {
        var images = new ImageClient();
        using var lifetime = new Lifetime();
        lifetime.StopApplication();
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            Create(images, lifetime, new DockerImageSettings(), new CLang()).ExecuteAsync());
        Assert.IsEmpty(images.Pulled);
    }

    private static RuntimeImageWarmupJob Create(ImageClient images, Lifetime lifetime,
        DockerImageSettings settings, params ILang[] languages)
    {
        var gpu = new HasGpuService(new CommandService(),
            new CacheService(new Microsoft.Extensions.Caching.Memory.MemoryCache(new Microsoft.Extensions.Caching.Memory.MemoryCacheOptions()),
                NullLogger<CacheService>.Instance), NullLogger<HasGpuService>.Instance);
        return new RuntimeImageWarmupJob(images, languages, gpu,
            new DockerRegistryLoginService(NullLogger<DockerRegistryLoginService>.Instance),
            Options.Create(settings), lifetime, NullLogger<RuntimeImageWarmupJob>.Instance);
    }

    internal sealed class ImageClient : IDockerImageClient
    {
        public HashSet<string> Cached { get; } = new(StringComparer.Ordinal);
        public ConcurrentBag<string> Pulled { get; } = [];
        public ConcurrentDictionary<string, bool> Prepared { get; } = new();
        public TaskCompletionSource TwoStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool Available { get; init; }
        public bool Block { get; init; }
        public string? FailImage { get; set; }
        public int Inspections;

        public Task<HashSet<string>> ListCachedAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(Cached.Concat(Prepared.Keys).ToHashSet(StringComparer.Ordinal));
        }

        public Task<bool> IsAvailableAsync(string image, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref Inspections);
            return Task.FromResult(Available || Cached.Contains(image) || Prepared.ContainsKey(image));
        }

        public async Task PullAsync(string image, CancellationToken cancellationToken)
        {
            Pulled.Add(image);
            if (Pulled.Count == 2) TwoStarted.TrySetResult();
            if (Block) await Task.Delay(Timeout.Infinite, cancellationToken);
            if (image == FailImage) throw new InvalidOperationException("Registry unavailable for this image.");
            Prepared[image] = true;
        }
    }

    internal sealed class Lifetime : IHostApplicationLifetime, IDisposable
    {
        private readonly CancellationTokenSource _stopping = new();
        public CancellationToken ApplicationStarted => CancellationToken.None;
        public CancellationToken ApplicationStopping => _stopping.Token;
        public CancellationToken ApplicationStopped => CancellationToken.None;
        public void StopApplication() => _stopping.Cancel();
        public void Dispose() => _stopping.Dispose();
    }

    private sealed class Language(string image) : ILang
    {
        public string LangDisplayName => "Test language";
        public string LangName => "test";
        public string LangExtension => "txt";
        public string DefaultCode => "";
        public string EntryFileName => "main.txt";
        public string DockerImage => image;
        public string RunCommand => "true";
        public bool NeedGpu => false;
        public Dictionary<string, string> OtherFiles => [];
    }
}
