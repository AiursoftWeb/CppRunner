using System.Net;
using Aiursoft.CppRunner.Services;
using Aiursoft.CppRunner.Services.FileStorage;

using Aiursoft.CppRunner.Entities;
using Microsoft.AspNetCore.Identity;
namespace Aiursoft.CppRunner.Tests.IntegrationTests;

[TestClass]
public class ManageControllerTests : TestBase
{
    [TestMethod]
    public async Task TestManageWorkflow()
    {
        await LoginAsAdmin();

        // Ensure AllowUserAdjustNickname is true
        using (var scope = Server!.Services.CreateScope())
        {
            var settingsService = scope.ServiceProvider.GetRequiredService<GlobalSettingsService>();
            await settingsService.UpdateSettingAsync(Configuration.SettingsMap.AllowUserAdjustNickname, "True");
        }

        // 1. Index
        var indexResponse = await Http.GetAsync("/Manage/Index");
        indexResponse.EnsureSuccessStatusCode();

        // 2. ChangePassword (GET)
        var changePasswordPage = await Http.GetAsync("/Manage/ChangePassword");
        changePasswordPage.EnsureSuccessStatusCode();

        // 3. ChangeProfile (GET)
        var changeProfilePage = await Http.GetAsync("/Manage/ChangeProfile");
        changeProfilePage.EnsureSuccessStatusCode();

        // 4. ChangeAvatar (GET)
        var changeAvatarPage = await Http.GetAsync("/Manage/ChangeAvatar");
        changeAvatarPage.EnsureSuccessStatusCode();
    }

    [TestMethod]
    public async Task TestChangePasswordFailure()
    {
        await RegisterAndLoginAsync();

        // Test with wrong old password
        var response = await PostForm("/Manage/ChangePassword", new Dictionary<string, string>
        {
            { "OldPassword", "WrongPassword" },
            { "NewPassword", "NewPassword123!" },
            { "ConfirmPassword", "NewPassword123!" }
        });

        Assert.AreEqual(HttpStatusCode.OK, response.StatusCode);
        var html = await response.Content.ReadAsStringAsync();
        Assert.Contains("Incorrect password.", html);
    }

    [TestMethod]
    public async Task TestChangeProfileFailure()
    {
        await RegisterAndLoginAsync();

        // Test with invalid model (empty name)
        var response = await PostForm("/Manage/ChangeProfile", new Dictionary<string, string>
        {
            { "Name", "" }
        });

        Assert.AreEqual(HttpStatusCode.OK, response.StatusCode);
        // Should stay on the same page with validation error
    }

    [TestMethod]
    public async Task TestChangeAvatarAllowsJpegExtension()
    {
        await RegisterAndLoginAsync();

        var response = await Http.GetAsync("/Manage/ChangeAvatar");
        response.EnsureSuccessStatusCode();

        var html = await response.Content.ReadAsStringAsync();
        Assert.Contains("data-allowed-file-extensions=\"png bmp jpg jpeg\"", html);
        Assert.Contains("validExtensions: ('png bmp jpg jpeg' || '').split(' ').filter(Boolean)", html);
    }

    [TestMethod]
    public async Task TestChangeAvatarInvalidImage()
    {
        await RegisterAndLoginAsync();

        // Upload a non-image file
        var content = new StringContent("Not an image");
        var multipartContent = new MultipartFormDataContent();
        multipartContent.Add(content, "file", "test.txt");

        var storage = GetService<StorageService>();
        var uploadUrl = storage.GetUploadUrl("avatar", isVault: false);
        var uploadResponse = await Http.PostAsync(uploadUrl, multipartContent);
        uploadResponse.EnsureSuccessStatusCode();
        var uploadResult = await uploadResponse.Content.ReadFromJsonAsync<UploadResult>();
        string path = uploadResult!.Path;

        var response = await PostForm("/Manage/ChangeAvatar", new Dictionary<string, string>
        {
            { "AvatarUrl", path }
        });

        Assert.AreEqual(HttpStatusCode.OK, response.StatusCode);
        var html = await response.Content.ReadAsStringAsync();
        Assert.Contains("The file is not a valid image.", html);
    }

    private class UploadResult
    {
        public string Path { get; init; } = string.Empty;
    }

    [TestMethod]
    public async Task TestDeleteAccount_Authenticated_DeletesUserAndSignsOut()
    {
        // Arrange: register and login as a new user
        var (email, _) = await RegisterAndLoginAsync();

        // Act: confirmation page loads
        var deletePage = await Http.GetAsync("/Manage/DeleteAccount");
        deletePage.EnsureSuccessStatusCode();

        // Act: confirm deletion
        var deleteResponse = await PostForm("/Manage/DeleteAccountPost", new(), tokenUrl: "/Manage/DeleteAccount");
        AssertRedirect(deleteResponse, "/");

        // Assert: signed out — accessing a protected page should redirect to login
        var managePage = await Http.GetAsync("/Manage/Index");
        Assert.AreEqual(HttpStatusCode.Found, managePage.StatusCode);

        // Assert: user no longer exists in the database
        using var scope = Server!.Services.CreateScope();
        var userManager = scope.ServiceProvider.GetRequiredService<UserManager<User>>();
        Assert.IsNull(await userManager.FindByEmailAsync(email));
    }

    [TestMethod]
    public async Task TestDeleteAccount_Unauthenticated_RedirectsToLogin()
    {
        // No login — direct access should redirect
        var deletePage = await Http.GetAsync("/Manage/DeleteAccount");
        Assert.AreEqual(HttpStatusCode.Found, deletePage.StatusCode);
    }
}
