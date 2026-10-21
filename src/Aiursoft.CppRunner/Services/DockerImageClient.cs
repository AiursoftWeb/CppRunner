using System.Diagnostics;

namespace Aiursoft.CppRunner.Services;

public interface IDockerImageClient
{
    Task<HashSet<string>> ListCachedAsync(CancellationToken cancellationToken);
    Task<bool> IsAvailableAsync(string image, CancellationToken cancellationToken);
    Task PullAsync(string image, CancellationToken cancellationToken);
}

public class DockerImageClient : IDockerImageClient
{
    public async Task<HashSet<string>> ListCachedAsync(CancellationToken cancellationToken)
    {
        var result = await ExecuteAsync(["image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            TimeSpan.FromSeconds(10), cancellationToken);
        if (result.Code != 0) throw new InvalidOperationException($"Cannot list runtime images: {result.Error}");
        return result.Output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToHashSet(StringComparer.Ordinal);
    }

    public async Task<bool> IsAvailableAsync(string image, CancellationToken cancellationToken)
    {
        var result = await ExecuteAsync(["image", "inspect", "--format", "{{.Id}}", image],
            TimeSpan.FromSeconds(10), cancellationToken);
        return result.Code == 0;
    }

    public async Task PullAsync(string image, CancellationToken cancellationToken)
    {
        var result = await ExecuteAsync(["pull", "--quiet", image], TimeSpan.FromMinutes(10), cancellationToken);
        if (result.Code != 0) throw new InvalidOperationException($"Cannot prepare runtime image {image}: {result.Error}");
    }

    private static async Task<(int Code, string Output, string Error)> ExecuteAsync(
        string[] arguments, TimeSpan timeout, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var process = new Process();
        process.StartInfo = new ProcessStartInfo("docker")
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true
        };
        foreach (var argument in arguments) process.StartInfo.ArgumentList.Add(argument);
        process.Start();
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(timeout);
        try
        {
            await process.WaitForExitAsync(deadline.Token);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync(CancellationToken.None);
            await Task.WhenAll(output, error);
            cancellationToken.ThrowIfCancellationRequested();
            throw new TimeoutException("Docker runtime image preparation timed out.");
        }
        return (process.ExitCode, await output, await error);
    }
}
