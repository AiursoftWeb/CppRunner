using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using Aiursoft.CSTools.Services;

namespace Aiursoft.CppRunner.Controllers;

public class HomeController : Controller
{
    private readonly string _tempFolder = Path.GetTempPath();
    private readonly CommandService _commandService;

    public HomeController(CommandService commandService)
    {
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
        // Entire posted from is C++ code.
        var content = await new StreamReader(Request.Body).ReadToEndAsync();
        var buildId = Guid.NewGuid().ToString("N");
        var folder = Path.Combine(_tempFolder, buildId);
        Directory.CreateDirectory(folder);
        
        var sourceFile = Path.Combine(folder, "main.cpp");
        await System.IO.File.WriteAllTextAsync(sourceFile, content);
        var processId = 0;
        
        try
        {
            var (code, output, error) = await _commandService.RunCommandAsync(
                bin: "docker",
                arg: $"run --rm --cpus=0.5 --memory=256m --network none -v {folder}:/app frolvlad/alpine-gxx sh -c \"g++ /app/main.cpp -o /tmp/main && /tmp/main\"",
                path: _tempFolder,
                timeout: TimeSpan.FromSeconds(10),
                i => processId = i);
            
            return Json(new
            {
                code,
                output,
                error
            });
        }
        catch (TimeoutException e)
        {
            // Kill the process.
            if (processId != 0)
            {
                var process = Process.GetProcessById(processId);
                process.Kill();
            }
            return BadRequest(e.Message);
        }
    }
}
