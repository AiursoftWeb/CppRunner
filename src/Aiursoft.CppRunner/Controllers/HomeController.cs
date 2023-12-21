using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Lang;
using Microsoft.AspNetCore.Mvc;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner.Controllers;

public class HomeController : Controller
{
    private readonly string _tempFolder = Path.GetTempPath();
    private readonly CanonQueue _queue;
    private readonly IEnumerable<ILang> _langs;
    private readonly ILogger<HomeController> _logger;
    private readonly CommandService _commandService;

    public HomeController(
        CanonQueue queue,
        IEnumerable<ILang> langs,
        ILogger<HomeController> logger,
        CommandService commandService)
    {
        _queue = queue;
        _langs = langs;
        _logger = logger;
        _commandService = commandService;
    }

    public IActionResult Index()
    {
        return View();
    }

    [Route("langs")]
    public IActionResult GetSupportedLangs()
    {
        return this.Ok(_langs.Select(l => new
        {
            l.LangName,
            l.LangExtension,
        }));
    }

    [Route("run")]
    [HttpPost]
    public async Task<IActionResult> Run([FromQuery][Required]string lang, [FromForm]string content)
    {
        var langImplement = _langs.FirstOrDefault(t => string.Equals(t.LangExtension, lang, StringComparison.CurrentCultureIgnoreCase));
        if (langImplement == null)
        {
            return BadRequest("Lang not found!");
        }
        
        var buildId = Guid.NewGuid().ToString("N");
        var folder = Path.Combine(_tempFolder, buildId);
        Directory.CreateDirectory(folder);
        
        _logger.LogInformation("Build ID: {BuildId}", buildId);
        var sourceFile = Path.Combine(folder, langImplement.FileName);
        await System.IO.File.WriteAllTextAsync(sourceFile, content);
        var processId = 0;

        try
        {
            var (code, output, error) = await _commandService.RunCommandAsync(
                bin: "docker",
                arg:
                $"run --rm --name {buildId} --cpus=0.5 --memory=256m --network none -v {folder}:/app {langImplement.DockerImage} sh -c \"{langImplement.RunCommand}\"",
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(10),
                i => processId = i);
            _logger.LogInformation("{Build} Code: {Code}", buildId, code);

            return Ok(new
            {
                code,
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
