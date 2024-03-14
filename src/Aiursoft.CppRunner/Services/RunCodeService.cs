using Aiursoft.Canon;
using Aiursoft.CppRunner.Models;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner.Services;

public class RunCodeService
{
    private readonly string _tempFolder = Path.Combine(Path.GetTempPath(), "cpprunner", "builds");
    private readonly ILogger<RunCodeService> _logger;
    private readonly CommandService _commandService;
    private readonly CanonQueue _queue;

    public RunCodeService(
        ILogger<RunCodeService> logger, 
        CommandService commandService,
        CanonQueue queue)
    {
        _logger = logger;
        _commandService = commandService;
        _queue = queue;
    }
    
    public async Task<CodeResult> RunCode(string code, ILang lang)
    {
        var buildId = Guid.NewGuid().ToString("N");
        var folder = Path.Combine(_tempFolder, buildId);
        Directory.CreateDirectory(folder);

        _logger.LogInformation("Build ID: {BuildId}", buildId);
        var sourceFile = Path.Combine(folder, lang.EntryFileName);
        await File.WriteAllTextAsync(sourceFile, code);

        foreach (var otherFile in lang.OtherFiles)
        {
            _logger.LogInformation("Writing file {FileName} to {Folder}", otherFile.Key, folder);
            await File.WriteAllTextAsync(Path.Combine(folder, otherFile.Key), otherFile.Value);
        }

        try
        {
            var (resultCode, output, error) = await _commandService.RunCommandAsync(
                bin: "docker",
                arg:
                $"run --rm --name {buildId} --cpus=8 --memory=512m --network none -v {folder}:/app {lang.DockerImage} sh -c \"{lang.RunCommand}\"",
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(30),
                killTimeoutProcess: true);
            _logger.LogInformation("{Build} Code: {Code}", buildId, resultCode);

            return new CodeResult
            {
                ResultCode = resultCode,
                Output = output,
                Error = error
            };
        }
        catch (TimeoutException e)
        {
            _logger.LogError(e, "Timeout with build {Build}", buildId);
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
            _logger.LogInformation("Killing container {Build}", buildId);
            _ = await _commandService.RunCommandAsync(
                bin: "docker",
                arg: $"kill {buildId}",
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(10));

            _logger.LogInformation("Removing container {Build}", buildId);
            _ = await _commandService.RunCommandAsync(
                bin: "docker",
                arg: $"rm -f {buildId}",
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(10));

            _logger.LogInformation("Removing folder {Build}", buildId);
            _queue.QueueNew(() =>
            {
                CSTools.Tools.FolderDeleter.DeleteByForce(folder);
                return Task.CompletedTask;
            });
        }
    }
}