using Aiursoft.CppRunner.Lang;
using Microsoft.AspNetCore.Mvc;

namespace Aiursoft.CppRunner.Controllers;

[Route("langs")]
public class LangsController : ControllerBase
{
    private readonly IEnumerable<ILang> _langs;

    public LangsController(IEnumerable<ILang> langs)
    {
        _langs = langs;
    }
    
    [Route("")]
    public IActionResult GetSupportedLangs()
    {
        return this.Ok(_langs.Select(l => new
        {
            l.LangName,
            l.LangDisplayName,
            l.LangExtension,
        }));
    }
    
    [Route("{lang}/default")]
    public IActionResult GetlangDefaultCode(string lang)
    {
        var langDetails = _langs.FirstOrDefault(l => string.Equals(l.LangName, lang, StringComparison.CurrentCultureIgnoreCase));
        if (langDetails == null)
        {
            return NotFound();
        }
        
        return this.Ok(langDetails.DefaultCode);
    }
}