using System.Diagnostics;
using Aiursoft.CppRunner.Configuration;

namespace Aiursoft.CppRunner.Services;

public class DockerRegistryLoginService(ILogger<DockerRegistryLoginService> logger)
{
    public async Task LoginIfRequiredAsync(
        DockerImageSettings settings,
        string dockerExecutable = "docker")
    {
        if (!settings.RequireAuthentication)
        {
            return;
        }

        var registry = GetRegistryFromPrefix(settings.Prefix);
        ArgumentException.ThrowIfNullOrWhiteSpace(settings.Username);
        ArgumentException.ThrowIfNullOrWhiteSpace(settings.Password);

        logger.LogInformation("Logging in to Docker registry {Registry} as {Username}.", registry, settings.Username);

        using var process = new Process();
        process.StartInfo = new ProcessStartInfo
        {
            FileName = dockerExecutable,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        process.StartInfo.ArgumentList.Add("login");
        process.StartInfo.ArgumentList.Add("--username");
        process.StartInfo.ArgumentList.Add(settings.Username);
        process.StartInfo.ArgumentList.Add("--password-stdin");
        process.StartInfo.ArgumentList.Add(registry);

        process.Start();
        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();

        await process.StandardInput.WriteLineAsync(settings.Password);
        process.StandardInput.Close();

        try
        {
            await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(30));
        }
        catch (TimeoutException)
        {
            process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync();
            throw new TimeoutException($"Docker login to registry '{registry}' timed out.");
        }

        await outputTask;
        var error = await errorTask;
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"Docker login to registry '{registry}' failed with exit code {process.ExitCode}: {error.Trim()}");
        }

        logger.LogInformation("Docker login to registry {Registry} succeeded.", registry);
    }

    public static string GetRegistryFromPrefix(string prefix)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefix);

        var trimmedPrefix = prefix.Trim().TrimEnd('/');
        if (trimmedPrefix.Contains("://", StringComparison.Ordinal))
        {
            throw new ArgumentException("Docker image prefix must not contain a URL scheme.", nameof(prefix));
        }

        var registry = trimmedPrefix.Split('/', 2)[0];
        if (!registry.Contains('.') && !registry.Contains(':') && registry != "localhost")
        {
            throw new ArgumentException(
                "Authenticated Docker image prefix must start with a registry host, for example 'hub.aiursoft.com/'.",
                nameof(prefix));
        }

        return registry;
    }
}
