using Aiursoft.CppRunner.Services;
using Aiursoft.CSTools.Tools;
using Microsoft.AspNetCore.Mvc;

namespace Aiursoft.CppRunner.Controllers;

[Route("{controller}")]
public class LangsController(
    HasGpuService hasGpuService,
    IEnumerable<ILang> langs) : ControllerBase
{
    [Route("")]
    public async Task<IActionResult> GetSupportedLangs()
    {
        var hasGpu = await hasGpuService.HasNvidiaGpuForDockerWithCache();
        var availableLangs = hasGpu
            ? langs
            : langs.Where(l => !l.NeedGpu);
        return Ok(availableLangs.Select(l => new
        {
            l.LangName,
            l.LangDisplayName,
            l.LangExtension,
        }));
    }

    [Route("{lang}/default")]
    public async Task<IActionResult> GetLangDefaultCode(string lang)
    {
        var hasGpu = await hasGpuService.HasNvidiaGpuForDockerWithCache();
        var availableLangs = langs;
        if (!hasGpu)
        {
            availableLangs = langs.Where(l => !l.NeedGpu);
        }

        return availableLangs.TryFindFirst<ILang, IActionResult>(l => string.Equals(l.LangName, lang, StringComparison.CurrentCultureIgnoreCase),
            onFound: langDetails => Ok(langDetails.DefaultCode),
            onNotFound: NotFound);
    }
}
