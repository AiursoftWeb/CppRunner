using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner.Controllers;

public class HomeController : Controller
{
    private readonly string _tempFolder = Path.GetTempPath();
    private readonly ILogger<HomeController> _logger;
    private readonly CommandService _commandService;

    public HomeController(
        ILogger<HomeController> logger,
        CommandService commandService)
    {
        _logger = logger;
        _commandService = commandService;
    }

    public IActionResult Index()
    {
        return Ok();
    }

    [Route("run")]
    [HttpPost]
    public async Task<IActionResult> Run()
    {
        // Entire posted form is C++ code.
        var content = await new StreamReader(Request.Body).ReadToEndAsync();
        var buildId = Guid.NewGuid().ToString("N");
        var folder = Path.Combine(_tempFolder, buildId);
        Directory.CreateDirectory(folder);
        
        _logger.LogInformation("Build ID: {BuildId}", buildId);
        var sourceFile = Path.Combine(folder, "main.cpp");
        await System.IO.File.WriteAllTextAsync(sourceFile, content);
        var processId = 0;

        try
        {
            var (code, output, error) = await _commandService.RunCommandAsync(
                bin: "docker",
                arg:
                $"run --rm --name {buildId} --cpus=0.5 --memory=256m --network none -v {folder}:/app frolvlad/alpine-gxx sh -c \"g++ /app/main.cpp -o /tmp/main && /tmp/main\"",
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

            return BadRequest(e.Message);
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
            CSTools.Tools.FolderDeleter.DeleteByForce(folder);
        }
    }
}
