using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Lang;
using Microsoft.AspNetCore.Mvc;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner.Controllers;

[Route("runner")]
public class RunnerController : ControllerBase
{
    private readonly string _tempFolder = Path.GetTempPath();
    private readonly CanonQueue _queue;
    private readonly IEnumerable<ILang> _langs;
    private readonly ILogger<RunnerController> _logger;
    private readonly CommandService _commandService;

    public RunnerController(
        CanonQueue queue,
        IEnumerable<ILang> langs,
        ILogger<RunnerController> logger,
        CommandService commandService)
    {
        _queue = queue;
        _langs = langs;
        _logger = logger;
        _commandService = commandService;
    }

    [Route("run")]
    [HttpPost]
    public async Task<IActionResult> Run([FromQuery][Required]string lang)
    {
        var langImplement = _langs.FirstOrDefault(t => string.Equals(t.LangExtension, lang, StringComparison.CurrentCultureIgnoreCase));
        if (langImplement == null)
        {
            return BadRequest("Lang not found!");
        }
        
        var code = await new StreamReader(Request.Body).ReadToEndAsync();
        
        var buildId = Guid.NewGuid().ToString("N");
        var folder = Path.Combine(_tempFolder, buildId);
        Directory.CreateDirectory(folder);
        
        _logger.LogInformation("Build ID: {BuildId}", buildId);
        var sourceFile = Path.Combine(folder, langImplement.FileName);
        await System.IO.File.WriteAllTextAsync(sourceFile, code);

        foreach (var otherFile in langImplement.OtherFiles)
        {
            _logger.LogInformation("Writing file {FileName} to {Folder}", otherFile.Key, folder);
            await System.IO.File.WriteAllTextAsync(Path.Combine(folder, otherFile.Key), otherFile.Value);
        }
        
        var processId = 0;
        try
        {
            var (resultCode, output, error) = await _commandService.RunCommandAsync(
                bin: "docker",
                arg:
                $"run --rm --name {buildId} --cpus=0.5 --memory=256m --network none -v {folder}:/app {langImplement.DockerImage} sh -c \"{langImplement.RunCommand}\"",
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(10),
                i => processId = i);
            _logger.LogInformation("{Build} Code: {Code}", buildId, resultCode);

            return Ok(new
            {
                resultCode,
                output,
                error
            });
        }
        catch (TimeoutException e)
        {
            _logger.LogError(e, "Timeout with build {Build}", buildId);
            // Kill the process.
            if (processId != 0)
            {
                _logger.LogInformation("Killing process {ProcessId}", processId);
                var process = Process.GetProcessById(processId);
                process.Kill();
            }

            return BadRequest("Timeout");
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
