using System.Collections.Concurrent;
using Aiursoft.Canon.BackgroundJobs;
using Aiursoft.CppRunner.Configuration;
using Microsoft.Extensions.Options;

namespace Aiursoft.CppRunner.Services.BackgroundJobs;

public class RuntimeImageWarmupJob(
    IDockerImageClient images,
    IEnumerable<ILang> languages,
    HasGpuService gpu,
    DockerRegistryLoginService registry,
    IOptions<DockerImageSettings> settings,
    IHostApplicationLifetime lifetime,
    ILogger<RuntimeImageWarmupJob> logger) : IBackgroundJob
{
    public string Name => "Runtime Image Warmup";
    public string Description => "Prepares missing language images without blocking the website. Retry failed preparation here.";

    public async Task ExecuteAsync()
    {
        var cancellationToken = lifetime.ApplicationStopping;
        var cached = await images.ListCachedAsync(cancellationToken);
        var hasGpu = languages.Any(language => language.NeedGpu) && await gpu.HasNvidiaGpuForDockerWithCache();
        var candidates = languages.Where(language => !language.NeedGpu || hasGpu)
            .Select(language => language.GetDockerImagePullEndpoint(settings.Value.Prefix))
            .Distinct(StringComparer.Ordinal);
        var missing = new List<string>();
        foreach (var image in candidates)
        {
            // Inspect also recognizes Docker's canonical aliases and digest references.
            if (!cached.Contains(image) && !await images.IsAvailableAsync(image, cancellationToken)) missing.Add(image);
        }
        if (missing.Count == 0) return;
        await registry.LoginIfRequiredAsync(settings.Value, cancellationToken: cancellationToken);
        var failures = new ConcurrentBag<string>();
        await Parallel.ForEachAsync(missing, new ParallelOptions
        {
            MaxDegreeOfParallelism = 2, CancellationToken = cancellationToken
        }, async (image, token) =>
        {
            try
            {
                logger.LogInformation("Preparing runtime image {Image}", image);
                await images.PullAsync(image, token);
            }
            catch (OperationCanceledException) when (token.IsCancellationRequested) { throw; }
            catch (Exception ex)
            {
                failures.Add(image);
                logger.LogWarning(ex, "Runtime image {Image} is not ready; other languages remain available", image);
            }
        });
        if (!failures.IsEmpty)
            throw new InvalidOperationException($"Runtime images are not ready: {string.Join(", ", failures.Order())}. Retry this job to prepare them.");
    }
}
