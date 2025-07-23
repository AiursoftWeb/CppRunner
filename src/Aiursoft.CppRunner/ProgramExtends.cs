using System.Diagnostics.CodeAnalysis;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Services;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner;

public static class ProgramExtends
{
    public static string GetDockerImagePullEndpoint(this ILang lang, string? prefixFromConfig)
    {
        var trimmedPrefix = string.Empty;
        if (!string.IsNullOrWhiteSpace(prefixFromConfig))
        {
            trimmedPrefix = prefixFromConfig.TrimEnd('/') + '/';
        }

        if (lang.DockerImage.StartsWith(trimmedPrefix))
        {
            return lang.DockerImage;
        }

        return trimmedPrefix + lang.DockerImage;
    }

    [ExcludeFromCodeCoverage]
    public static async Task PullContainersAsync(this IHost host)
    {
        using var scope = host.Services.CreateScope();
        var services = scope.ServiceProvider;
        var commandService = services.GetRequiredService<CommandService>();
        var configuration = services.GetRequiredService<IConfiguration>();
        var logger = services.GetRequiredService<ILogger<Startup>>();
        var langs = services.GetRequiredService<IEnumerable<ILang>>();
        var hasGpuService = services.GetRequiredService<HasGpuService>();
        var retryEngine = services.GetRequiredService<RetryEngine>();
        var pool = services.GetRequiredService<CanonPool>();
        var prefix = configuration["DockerImageSettings:Prefix"];

        var downloadedImages = await commandService.RunCommandAsync("docker", "images", Path.GetTempPath());
        logger.LogInformation("Downloaded images count: {ImagesCount}", downloadedImages.output.Split('\n').Length);

        var hasGpu = await hasGpuService.HasNvidiaGpuForDockerWithCache();
        logger.LogInformation("Has GPU: {HasGpu}", hasGpu);

        var availableLangs = hasGpu
            ? langs
            : langs.Where(l => !l.NeedGpu);

        foreach (var lang in availableLangs)
        {
            if (downloadedImages.output.Contains(lang.GetDockerImagePullEndpoint(prefix)))
            {
                logger.LogInformation("Docker image {Image} already downloaded.", lang.GetDockerImagePullEndpoint(prefix));
                continue;
            }
            pool.RegisterNewTaskToPool(() => retryEngine.RunWithRetry(async _ =>
            {
                logger.LogInformation("Pulling docker image {Image}", lang.GetDockerImagePullEndpoint(prefix));
                var result = await commandService.RunCommandAsync("docker", $"pull {lang.GetDockerImagePullEndpoint(prefix)}", path: Path.GetTempPath(), timeout: TimeSpan.FromMinutes(10));
                if (result.code != 0)
                {
                    throw new Exception($"Failed to pull docker image {lang.GetDockerImagePullEndpoint(prefix)}! Error: {result.error}");
                }
            }, 5));
        }

        await pool.RunAllTasksInPoolAsync();
    }
}
