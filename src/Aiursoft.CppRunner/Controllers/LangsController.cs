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
        return Ok(_langs.Select(l => new
        {
            l.LangName,
            l.LangDisplayName,
            l.LangExtension,
        }));
    }
    
    [Route("{lang}/default")]
    public IActionResult GetLangDefaultCode(string lang)
    {
        var langDetails = _langs.SingleOrDefault(l => string.Equals(l.LangName, lang, StringComparison.CurrentCultureIgnoreCase));
        if (langDetails == null)
        {
            return NotFound();
        }
        
        return Ok(langDetails.DefaultCode);
    }
}