using Aiursoft.CppRunner.Entities;
using Aiursoft.CppRunner.Models.HomeViewModels;
using Aiursoft.CppRunner.Services;
using Aiursoft.UiStack.Navigation;
using Aiursoft.WebTools.Attributes;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Aiursoft.CppRunner.Controllers;

[LimitPerMin]
public class HomeController(
    HasGpuService hasGpuService,
    IEnumerable<ILang> langs,
    UserManager<User> userManager,
    TemplateDbContext dbContext) : Controller
{
    [RenderInNavBar(
        NavGroupName = "Features",
        NavGroupOrder = 0,
        CascadedLinksGroupName = "Runner",
        CascadedLinksIcon = "terminal",
        CascadedLinksOrder = 0,
        LinkText = "Code Runner",
        LinkOrder = 0)]
    public async Task<IActionResult> Index(int? codeId = null)
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

        if (codeId.HasValue)
        {
            var savedCode = await dbContext.SavedCodes
                .Include(c => c.User)
                .FirstOrDefaultAsync(c => c.Id == codeId.Value);

            if (savedCode != null)
            {
                var canRead = savedCode.IsPublic;
                if (!canRead)
                {
                    var user = await userManager.GetUserAsync(User);
                    if (user != null && user.Id == savedCode.UserId)
                    {
                        canRead = true;
                    }
                }

                if (canRead)
                {
                    vm.PreLoadedCode = savedCode.Code;
                    vm.PreLoadedLang = savedCode.Language;
                    vm.PreLoadedTitle = savedCode.Title;
                }
            }
        }

        return this.StackView(vm);
    }

    [RenderInNavBar(
        NavGroupName = "Features",
        NavGroupOrder = 0,
        CascadedLinksGroupName = "Runner",
        CascadedLinksIcon = "terminal",
        CascadedLinksOrder = 0,
        LinkText = "Self Host",
        LinkOrder = 1)]
    public IActionResult SelfHost()
    {
        var vm = new SelfHostViewModel();
        return this.StackView(vm);
    }
}
