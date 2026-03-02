using Aiursoft.CppRunner.Models.HomeViewModels;
using Aiursoft.CppRunner.Services;
using Aiursoft.UiStack.Navigation;
using Aiursoft.WebTools.Attributes;
using Microsoft.AspNetCore.Mvc;

namespace Aiursoft.CppRunner.Controllers;

[LimitPerMin]
public class HomeController(
    HasGpuService hasGpuService,
    IEnumerable<ILang> langs) : Controller
{
    [RenderInNavBar(
        NavGroupName = "Features",
        NavGroupOrder = 0,
        CascadedLinksGroupName = "Runner",
        CascadedLinksIcon = "terminal",
        CascadedLinksOrder = 0,
        LinkText = "Code Runner",
        LinkOrder = 0)]
    public async Task<IActionResult> Index()
    {
        var hasGpu = await hasGpuService.HasNvidiaGpuForDockerWithCache();
        var availableLangs = (hasGpu ? langs : langs.Where(l => !l.NeedGpu)).ToList();

        var vm = new IndexViewModel
        {
            Langs = availableLangs.Select(l => new LangInfo
            {
                LangName = l.LangName,
                LangDisplayName = l.LangDisplayName,
                LangExtension = l.LangExtension,
                DefaultCode = l.DefaultCode
            }).ToList()
        };

        return this.StackView(vm);
    }
}
