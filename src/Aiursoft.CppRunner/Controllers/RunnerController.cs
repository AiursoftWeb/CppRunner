using Aiursoft.CppRunner.Services;
using Aiursoft.CSTools.Tools;
using Microsoft.AspNetCore.Mvc;
using Aiursoft.WebTools.Attributes;

namespace Aiursoft.CppRunner.Controllers;

[Route("runner")]
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
                return await RenderRunLangResult(langImplement, code);
            },
            onNotFound: () => Task.FromResult(NotFound() as IActionResult));
    
    private async Task<IActionResult> RenderRunLangResult(ILang langImplement, string code)
    {
        var result = await runCodeService.RunCode(code, langImplement);
        return Ok(result);
    }
}