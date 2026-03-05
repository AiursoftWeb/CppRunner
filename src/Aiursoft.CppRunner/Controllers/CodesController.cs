using Aiursoft.CppRunner.Entities;
using Aiursoft.CppRunner.Services;
using Aiursoft.WebTools.Attributes;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Aiursoft.CppRunner.Models.CodesViewModels;

namespace Aiursoft.CppRunner.Controllers;

[Route("{controller}")]
public class CodesController(
    UserManager<User> userManager,
    TemplateDbContext dbContext)
    : Controller
{
    [HttpGet]
    [Route("Index")]
    [Authorize]
    [Aiursoft.UiStack.Navigation.RenderInNavBar(
        NavGroupName = "Features",
        NavGroupOrder = 0,
        CascadedLinksGroupName = "Code Library",
        CascadedLinksIcon = "library",
        CascadedLinksOrder = 1,
        LinkText = "My Codes",
        LinkOrder = 0)]
    public async Task<IActionResult> Index()
    {
        var user = await userManager.GetUserAsync(User);
        if (user == null) return Unauthorized();

        var myCodes = await dbContext.SavedCodes
            .Where(c => c.UserId == user.Id)
            .OrderByDescending(c => c.CreationTime)
            .ToListAsync();

        var model = new IndexViewModel
        {
            MyCodes = myCodes
        };
        return this.StackView(model);
    }

    [HttpGet]
    [Route("Public")]
    [Aiursoft.UiStack.Navigation.RenderInNavBar(
        NavGroupName = "Features",
        NavGroupOrder = 0,
        CascadedLinksGroupName = "Code Library",
        CascadedLinksIcon = "library",
        CascadedLinksOrder = 1,
        LinkText = "Public Codes",
        LinkOrder = 1)]
    public async Task<IActionResult> Public()
    {
        var publicCodes = await dbContext.SavedCodes
            .Include(c => c.User)
            .Where(c => c.IsPublic)
            .OrderByDescending(c => c.CreationTime)
            .ToListAsync();

        var model = new PublicViewModel
        {
            PublicCodes = publicCodes
        };
        return this.StackView(model);
    }

    [HttpPost]
    [Route("Save")]
    [Authorize]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Save(string title, string code, string language, bool isPublic)
    {
        var user = await userManager.GetUserAsync(User);
        if (user == null) return Unauthorized();

        if (string.IsNullOrWhiteSpace(title))
        {
            ModelState.AddModelError(nameof(title), "Title is required.");
            return BadRequest(ModelState);
        }

        var savedCode = new SavedCode
        {
            UserId = user.Id,
            User = user,
            Title = title,
            Code = code,
            Language = language,
            IsPublic = isPublic,
            CreationTime = DateTime.UtcNow
        };

        dbContext.SavedCodes.Add(savedCode);
        await dbContext.SaveChangesAsync();

        return Ok(new { id = savedCode.Id });
    }

    [HttpGet]
    [Route("Read/{id:int}")]
    public async Task<IActionResult> Read(int id)
    {
        var savedCode = await dbContext.SavedCodes
            .Include(c => c.User)
            .FirstOrDefaultAsync(c => c.Id == id);

        if (savedCode == null)
        {
            return NotFound();
        }

        if (!savedCode.IsPublic)
        {
            var user = await userManager.GetUserAsync(User);
            if (user == null || user.Id != savedCode.UserId)
            {
                return Unauthorized();
            }
        }

        // Redirect to home with the code loaded.
        // We'll use TempData or just pass IDs to Home/Index
        return RedirectToAction("Index", "Home", new { codeId = id });
    }

    [HttpPost]
    [Route("Delete/{id:int}")]
    [Authorize]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Delete(int id)
    {
        var user = await userManager.GetUserAsync(User);
        if (user == null) return Unauthorized();

        var savedCode = await dbContext.SavedCodes
            .FirstOrDefaultAsync(c => c.Id == id && c.UserId == user.Id);

        if (savedCode == null)
        {
            return NotFound();
        }

        dbContext.SavedCodes.Remove(savedCode);
        await dbContext.SaveChangesAsync();

        return RedirectToAction(nameof(Index));
    }

    [HttpPost]
    [Route("TogglePublic/{id:int}")]
    [Authorize]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> TogglePublic(int id)
    {
        var user = await userManager.GetUserAsync(User);
        if (user == null) return Unauthorized();

        var savedCode = await dbContext.SavedCodes
            .FirstOrDefaultAsync(c => c.Id == id && c.UserId == user.Id);

        if (savedCode == null)
        {
            return NotFound();
        }

        savedCode.IsPublic = !savedCode.IsPublic;
        await dbContext.SaveChangesAsync();

        return RedirectToAction(nameof(Index));
    }
}
