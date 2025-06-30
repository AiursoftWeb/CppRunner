using System.Diagnostics;
using System.Text.RegularExpressions;
using Aiursoft.Canon;

namespace Aiursoft.CppRunner.Services;

public class HasGpuService(
    CacheService cacheService,
    ILogger<HasGpuService> logger)
{
    private static readonly Regex CliUuidRegex;

    static HasGpuService()
    {
        CliUuidRegex = new Regex(@"GPU UUID:\s*(GPU-[0-9A-Fa-f\-]+)", RegexOptions.Compiled);
    }

    public async Task<bool> HasNvidiaGpuForDockerWithCache()
    {
        return await cacheService.RunWithCache("HasNvidiaGpuForDocker",
            async () => await HasNvidiaGpuForDocker(),
            cachedMinutes: _ => TimeSpan.FromDays(3));
    }

    private async Task<bool> HasNvidiaGpuForDocker()
    {
        var lsPciHasNvidia = await LsPciHasNvidia();
        var nvidiaSmiReady = await NvidiaSmiReady();
        var hasNvidiaContainerToolkit = await HasNvidiaContainerToolkit();
        var isGpuUuidConsistent = await IsGpuUuidConsistent();
        var isDockerRuntimeNvidia = await IsDockerRuntimeNvidia();
        var finalResult =
            (lsPciHasNvidia &&
             nvidiaSmiReady &&
             hasNvidiaContainerToolkit &&
             isGpuUuidConsistent) || isDockerRuntimeNvidia;
        logger.LogInformation(
            "HasNvidiaGpuForDocker: {Result}, because: lspci has NVIDIA: {LsPciHasNvidia}, nvidia-smi is ready: {NvidiaSmiReady}, has nvidia-container-toolkit: {HasNvidiaContainerToolkit}, GPU UUIDs are consistent: {IsGpuUuidConsistent}",
            finalResult, lsPciHasNvidia, nvidiaSmiReady, hasNvidiaContainerToolkit, isGpuUuidConsistent);

        if (!lsPciHasNvidia)
        {
            logger.LogWarning("lspci has no NVIDIA. Can you make sure you have an NVIDIA GPU installed?");
        }
        if (!nvidiaSmiReady)
        {
            logger.LogWarning("nvidia-smi is not ready. Can you make sure NVIDIA drivers are installed and working?");
        }
        if (!hasNvidiaContainerToolkit)
        {
            logger.LogWarning("nvidia-container-toolkit is not installed. Can you make sure it is installed and configured?");
        }
        if (!isGpuUuidConsistent)
        {
            logger.LogWarning("GPU UUIDs are not consistent between nvidia-smi and nvidia-container-cli. This may cause issues with GPU access.");
        }
        if (!isDockerRuntimeNvidia)
        {
            logger.LogWarning("Docker runtime is not set to NVIDIA. This may cause issues with GPU access in Docker containers.");
        }
        return finalResult;
    }

    private async Task<bool> LsPciHasNvidia()
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "lspci",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            return string.IsNullOrEmpty(error) && output.Contains("NVIDIA");
        }
        catch
        {
            return false;
        }
    }

    private async Task<bool> NvidiaSmiReady()
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "nvidia-smi",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            return string.IsNullOrEmpty(error) && !string.IsNullOrEmpty(output) && output.Contains("NVIDIA-SMI") && process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private async Task<bool> HasNvidiaContainerToolkit()
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "nvidia-container-cli",
                    Arguments = "info",
                    RedirectStandardOutput = false,
                    RedirectStandardError = false,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            await process.WaitForExitAsync();

            return process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private async Task<string> GetGpuUuidFromNvidiaSmi()
    {
        var psi = new ProcessStartInfo
        {
            FileName = "nvidia-smi",
            Arguments = "--query-gpu=uuid --format=csv,noheader",
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var proc = Process.Start(psi)
                         ?? throw new InvalidOperationException("Cannot start nvidia-smi");
        var output = await proc.StandardOutput.ReadToEndAsync();
        await proc.WaitForExitAsync();

        // 取第一行，去除空白
        return output
                   .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                   .Select(line => line.Trim())
                   .FirstOrDefault()
               ?? throw new InvalidOperationException("Failed to read GPU UUID from nvidia-smi output");
    }

    private async Task<string> GetGpuUuidFromNvidiaContainerCli()
    {
        var psi = new ProcessStartInfo
        {
            FileName = "nvidia-container-cli",
            Arguments = "info",
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var proc = Process.Start(psi)
                         ?? throw new InvalidOperationException("Cannot start nvidia-container-cli");
        var output = await proc.StandardOutput.ReadToEndAsync();
        await proc.WaitForExitAsync();

        foreach (var line in output.Split('\n'))
        {
            var m = CliUuidRegex.Match(line);
            if (m.Success)
                return m.Groups[1].Value;
        }

        throw new InvalidOperationException("Cannot find GPU UUID in nvidia-container-cli output");
    }

    private async Task<bool> IsGpuUuidConsistent()
    {
        try
        {
            var uuidFromSmi = await GetGpuUuidFromNvidiaSmi();
            var uuidFromCli = await GetGpuUuidFromNvidiaContainerCli();
            return string.Equals(uuidFromSmi, uuidFromCli, StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to check GPU UUID consistency.");
            return false;
        }
    }

    private async Task<bool> IsDockerRuntimeNvidia()
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "docker",
                    Arguments = "info --format '{{.DefaultRuntime}}'",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            return string.IsNullOrEmpty(error) && output.Contains("nvidia") && process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }
}
