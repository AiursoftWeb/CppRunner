using Aiursoft.CppRunner.Services;
using Microsoft.AspNetCore.Mvc;
using Aiursoft.WebTools.Attributes;

namespace Aiursoft.CppRunner.Controllers;

[Route("runner")]
public class RunnerController : ControllerBase
{
    private readonly RunCodeService _runCodeService;
    private readonly IEnumerable<ILang> _langs;

    public RunnerController(
        RunCodeService runCodeService,
        IEnumerable<ILang> langs)
    {
        _runCodeService = runCodeService;
        _langs = langs;
    }

    [Route("run")]
    [HttpPost]
    [LimitPerMin(15)]
    public async Task<IActionResult> Run([FromQuery] string lang)
    {
        var langImplement =
            _langs.FirstOrDefault(t => string.Equals(t.LangName, lang, StringComparison.CurrentCultureIgnoreCase));
        if (langImplement == null)
        {
            return NotFound("Lang not found!");
        }

        var code = await new StreamReader(Request.Body).ReadToEndAsync();
        var result = await _runCodeService.RunCode(code, langImplement);
        return Ok(result);
    }
}