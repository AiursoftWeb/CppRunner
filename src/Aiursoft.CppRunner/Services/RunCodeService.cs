using Aiursoft.Canon;
using Aiursoft.CSTools.Services;
using Aiursoft.CppRunner.Models;
using Microsoft.Extensions.Localization;

namespace Aiursoft.CppRunner.Services;

public class RunCodeService(
    IConfiguration configuration,
    HasGpuService hasGpuService,
    ILogger<RunCodeService> logger,
    CommandService commandService,
    CanonQueue queue,
    IDockerImageClient images,
    IHostApplicationLifetime lifetime,
    IStringLocalizer<RunCodeService> localizer,
    ExecutionQueue executionQueue,
    IHttpContextAccessor httpContextAccessor)
{
    private readonly string _tempFolder = Path.Combine(Path.GetTempPath(), "cpprunner", "builds");

    private readonly string? _prefix = configuration["DockerImageSettings:Prefix"];

    public async Task<CodeResult> RunCode(string code, ILang lang, CancellationToken cancellationToken = default)
    {
        if (lang.NeedGpu && httpContextAccessor.HttpContext?.User.Identity?.IsAuthenticated != true)
        {
            return new CodeResult
            {
                ResultCode = StatusCodes.Status401Unauthorized,
                Error = localizer["Sign in to run code that requires a GPU."].Value
            };
        }

        using var linkedCancellation = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken, httpContextAccessor.HttpContext?.RequestAborted ?? CancellationToken.None,
            lifetime.ApplicationStopping);
        using var lease = await executionQueue.AcquireAsync(linkedCancellation.Token);
        linkedCancellation.Token.ThrowIfCancellationRequested();
        if (!lease.IsAcquired)
        {
            throw new InvalidOperationException("The execution queue is unavailable.");
        }
        var hasGpu = lang.NeedGpu && await hasGpuService.HasNvidiaGpuForDockerWithCache();
        var needGpu = lang.NeedGpu;
        if (needGpu && !hasGpu)
        {
            logger.LogWarning("Requested to run code with GPU, but no GPU is available.");
            return new CodeResult
            {
                ResultCode = 1,
                Output = "No GPU available!",
                Error = "No GPU available!"
            };
        }

        if (!await images.IsAvailableAsync(lang.GetDockerImagePullEndpoint(_prefix), linkedCancellation.Token))
        {
            var message = localizer["The runtime image for {0} is not ready. Please retry after image warmup completes.", lang.LangDisplayName].Value;
            return new CodeResult { ResultCode = 75, Output = message, Error = message };
        }

        var buildId = Guid.NewGuid().ToString("N");
        var folder = Path.Combine(_tempFolder, buildId);
        Directory.CreateDirectory(folder);

        logger.LogInformation("Build ID: {BuildId}", buildId);
        var sourceFile = Path.Combine(folder, lang.EntryFileName);
        await File.WriteAllTextAsync(sourceFile, code);

        foreach (var otherFile in lang.OtherFiles)
        {
            logger.LogInformation("Writing file {FileName} to {Folder}", otherFile.Key, folder);
            await File.WriteAllTextAsync(Path.Combine(folder, otherFile.Key), otherFile.Value);
        }

        try
        {
            var securityOptions = "--cap-drop=ALL --cap-add=CHOWN --cap-add=SETUID --cap-add=SETGID";
            var command = lang.NeedGpu ?
                $"run --pull=never --rm --name {buildId} {securityOptions} --gpus all --cpus=8 --memory=512m --network none -v {folder}:/app {lang.GetDockerImagePullEndpoint(_prefix)} sh -c \"{lang.RunCommand}\"" :
                $"run --pull=never --rm --name {buildId} {securityOptions}            --cpus=8 --memory=512m --network none -v {folder}:/app {lang.GetDockerImagePullEndpoint(_prefix)} sh -c \"{lang.RunCommand}\"";
            var (resultCode, output, error) = await commandService.RunCommandAsync(
                bin: "docker",
                arg: command,
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(30),
                killTimeoutProcess: true);
            logger.LogInformation("{Build} Code: {Code}", buildId, resultCode);

            return new CodeResult
            {
                ResultCode = resultCode,
                Output = output,
                Error = error
            };
        }
        catch (TimeoutException e)
        {
            logger.LogError(e, "Timeout with build {Build}", buildId);
            return new CodeResult
            {
                ResultCode = 124,
                Output = "Timeout!",
                Error = e.Message
            };
        }
        finally
        {
            // Kill and remove the container.
            logger.LogInformation("Killing container {Build}", buildId);
            _ = await commandService.RunCommandAsync(
                bin: "docker",
                arg: $"kill {buildId}",
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(10));

            logger.LogInformation("Removing container {Build}", buildId);
            _ = await commandService.RunCommandAsync(
                bin: "docker",
                arg: $"rm -f {buildId}",
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(10));

            logger.LogInformation("Removing folder {Build}", buildId);
            queue.QueueNew(() =>
            {
                CSTools.Tools.FolderDeleter.DeleteByForce(folder);
                return Task.CompletedTask;
            });
        }
    }
}
