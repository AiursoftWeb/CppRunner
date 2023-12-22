using Aiursoft.Canon;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner;

public static class ProgramExtends
{
    public static async Task<IHost> PullContainersAsync(this IHost host)
    {
        using var scope = host.Services.CreateScope();
        var services = scope.ServiceProvider;
        var commandService = services.GetRequiredService<CommandService>();
        var logger = services.GetRequiredService<ILogger<Startup>>();
        var langs = services.GetRequiredService<IEnumerable<ILang>>();
        var retryEngine = services.GetRequiredService<RetryEngine>();

        var downloadedImages = await commandService.RunCommandAsync("docker", "images", Path.GetTempPath());
        
        foreach (var lang in langs)
        {
            if (downloadedImages.output.Contains(lang.DockerImage))
            {
                logger.LogInformation("Docker image {Image} already downloaded.", lang.DockerImage);
                continue;
            }
            
            await retryEngine.RunWithRetry(async _ =>
            {
                logger.LogInformation("Pulling docker image {Image}", lang.DockerImage);
                var result = await commandService.RunCommandAsync("docker", $"pull {lang.DockerImage}", path: Path.GetTempPath(), timeout: TimeSpan.FromMinutes(5));
                if (result.code != 0)
                {
                    throw new Exception($"Failed to pull docker image {lang.DockerImage}! Error: {result.error}");
                }
            }, 5);
        }
        return host;
    }
}