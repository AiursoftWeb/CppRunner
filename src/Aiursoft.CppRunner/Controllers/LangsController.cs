using Aiursoft.CSTools.Tools;
using Microsoft.AspNetCore.Mvc;

namespace Aiursoft.CppRunner.Controllers;

[Route("{controller}")]
public class LangsController(IEnumerable<ILang> langs) : ControllerBase
{
    [Route("")]
    public IActionResult GetSupportedLangs()
    {
        return Ok(langs.Select(l => new
        {
            l.LangName,
            l.LangDisplayName,
            l.LangExtension,
        }));
    }

    [Route("{lang}/default")]
    public IActionResult GetLangDefaultCode(string lang)
    {
        return langs.TryFindFirst<ILang, IActionResult>(l => string.Equals(l.LangName, lang, StringComparison.CurrentCultureIgnoreCase),
            onFound: langDetails => Ok(langDetails.DefaultCode),
            onNotFound: NotFound);
    }
}