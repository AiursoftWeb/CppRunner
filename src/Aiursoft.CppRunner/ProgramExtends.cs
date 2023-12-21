using Aiursoft.CppRunner.Lang;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner;

public static class ProgramExtends
{
    public static async Task<IHost> SeedAsync(this IHost host)
    {
        using var scope = host.Services.CreateScope();
        var services = scope.ServiceProvider;
        var commandService = services.GetRequiredService<CommandService>();
        var logger = services.GetRequiredService<ILogger<Startup>>();
        var langs = services.GetRequiredService<IEnumerable<ILang>>();

        foreach (var lang in langs)
        {
            logger.LogInformation("Pulling docker image {Image}", lang.DockerImage);
            await commandService.RunCommandAsync("docker", $"pull {lang.DockerImage}", path: Path.GetTempPath());
        }
        return host;
    }
}