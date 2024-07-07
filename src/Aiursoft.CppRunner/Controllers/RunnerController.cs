using Aiursoft.CppRunner.Services;
using Aiursoft.CSTools.Tools;
using Microsoft.AspNetCore.Mvc;
using Aiursoft.WebTools.Attributes;

namespace Aiursoft.CppRunner.Controllers;

[Route("{controller}")]
public class RunnerController(
    RunCodeService runCodeService,
    IEnumerable<ILang> langs)
    : ControllerBase
{
    [Route("run")]
    [HttpPost]
    [LimitPerMin(15)]
    public Task<IActionResult> Run([FromQuery] string lang) =>
        langs.TryFindFirst<ILang, Task<IActionResult>>(
            t => string.Equals(t.LangName, lang, StringComparison.CurrentCultureIgnoreCase),
            onFound: async langImplement =>
            {
                var code = await new StreamReader(Request.Body).ReadToEndAsync();
                var result = await runCodeService.RunCode(code, langImplement);
                return Ok(result);
            },
            onNotFound: () => Task.FromResult(NotFound() as IActionResult));
}