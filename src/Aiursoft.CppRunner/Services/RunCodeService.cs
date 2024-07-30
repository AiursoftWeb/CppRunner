using Aiursoft.Canon;
using Aiursoft.CppRunner.Models;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner.Services;

public class RunCodeService(
    ILogger<RunCodeService> logger,
    CommandService commandService,
    CanonQueue queue)
{
    private readonly string _tempFolder = Path.Combine(Path.GetTempPath(), "cpprunner", "builds");

    public async Task<CodeResult> RunCode(string code, ILang lang)
    {
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
            var command = lang.NeedGpu ? 
                $"run --rm --name {buildId} --gpus all --cpus=1 --memory=512m --network none -v {folder}:/app {lang.DockerImage} sh -c \"{lang.RunCommand}\"" : 
                $"run --rm --name {buildId}            --cpus=1 --memory=512m --network none -v {folder}:/app {lang.DockerImage} sh -c \"{lang.RunCommand}\"";
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