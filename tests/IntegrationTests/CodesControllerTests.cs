using System.Net;
using Aiursoft.CppRunner.Entities;
using Microsoft.EntityFrameworkCore;

namespace Aiursoft.CppRunner.Tests.IntegrationTests;

[TestClass]
public class CodesControllerTests : TestBase
{
    [TestMethod]
    public async Task TestSaveAndReadCode()
    {
        await RegisterAndLoginAsync();

        // Save code
        var saveResponse = await PostForm("/Codes/Save", new Dictionary<string, string>
        {
            { "title", "Test Code" },
            { "code", "int main() { return 0; }" },
            { "language", "cpp" },
            { "isPublic", "true" }
        });
        Assert.AreEqual(HttpStatusCode.OK, saveResponse.StatusCode);

        // Verify in DB
        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            var savedCode = await db.SavedCodes.FirstOrDefaultAsync(c => c.Title == "Test Code");
            Assert.IsNotNull(savedCode);
            Assert.AreEqual("int main() { return 0; }", savedCode.Code);
            Assert.AreEqual("cpp", savedCode.Language);
            Assert.IsTrue(savedCode.IsPublic);
        }

        // List codes
        var listResponse = await Http.GetAsync("/Codes/Index");
        Assert.AreEqual(HttpStatusCode.OK, listResponse.StatusCode);
        var content = await listResponse.Content.ReadAsStringAsync();
        Assert.IsTrue(content.Contains("Test Code"));

        // Read code (redirects to Home/Index)
        int id;
        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            id = (await db.SavedCodes.FirstAsync(c => c.Title == "Test Code")).Id;
        }

        var readResponse = await Http.GetAsync($"/Codes/Read/{id}");
        Assert.AreEqual(HttpStatusCode.Found, readResponse.StatusCode);
        Assert.IsTrue(readResponse.Headers.Location!.OriginalString.Contains($"codeId={id}"));

        // Verify Home/Index loads the code
        var homeResponse = await Http.GetAsync(readResponse.Headers.Location!.OriginalString);
        Assert.AreEqual(HttpStatusCode.OK, homeResponse.StatusCode);
        var homeContent = await homeResponse.Content.ReadAsStringAsync();
        Assert.IsTrue(homeContent.Contains("int main() { return 0; }"));
        // Check for the title inside the serialized JSON or similar.
        // Since it's in a JS script tag, it should be present.
        Assert.IsTrue(homeContent.Contains("Test Code"));
    }

    [TestMethod]
    public async Task TestPublicCodeVisibility()
    {
        await RegisterAndLoginAsync();

        // Save a public code
        await PostForm("/Codes/Save", new Dictionary<string, string>
        {
            { "title", "Public Code" },
            { "code", "print('hello')" },
            { "language", "python" },
            { "isPublic", "true" }
        });

        // Save a private code
        await PostForm("/Codes/Save", new Dictionary<string, string>
        {
            { "title", "Private Code" },
            { "code", "secret code" },
            { "language", "python" },
            { "isPublic", "false" }
        });

        // Guest can see public code in Public library
        var handler = new HttpClientHandler { AllowAutoRedirect = false };
        var guestHttp = new HttpClient(handler) { BaseAddress = Http.BaseAddress };
        var publicResponse = await guestHttp.GetAsync("/Codes/Public");
        Assert.AreEqual(HttpStatusCode.OK, publicResponse.StatusCode);
        var publicContent = await publicResponse.Content.ReadAsStringAsync();
        Assert.IsTrue(publicContent.Contains("Public Code"));
        Assert.IsFalse(publicContent.Contains("Private Code"));

        // Get actual IDs from the DB to be safe
        int publicId, privateId;
        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            publicId = (await db.SavedCodes.FirstAsync(c => c.Title == "Public Code")).Id;
            privateId = (await db.SavedCodes.FirstAsync(c => c.Title == "Private Code")).Id;
        }

        // Guest can read public code
        var readResponse = await guestHttp.GetAsync($"/Codes/Read/{publicId}");
        Assert.AreEqual(HttpStatusCode.Found, readResponse.StatusCode);

        // Guest cannot read private code
        var readPrivateResponse = await guestHttp.GetAsync($"/Codes/Read/{privateId}");
        Assert.AreEqual(HttpStatusCode.Unauthorized, readPrivateResponse.StatusCode);
    }

    [TestMethod]
    public async Task TestToggleVisibilityAndDelete()
    {
        await RegisterAndLoginAsync();

        // Save code
        await PostForm("/Codes/Save", new Dictionary<string, string>
        {
            { "title", "Toggle Test" },
            { "code", "code" },
            { "language", "cpp" },
            { "isPublic", "false" }
        });

        int id;
        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            id = (await db.SavedCodes.FirstAsync(c => c.Title == "Toggle Test")).Id;
        }

        // Toggle to public
        var toggleResponse = await PostForm($"/Codes/TogglePublic/{id}", new Dictionary<string, string>());
        Assert.AreEqual(HttpStatusCode.Found, toggleResponse.StatusCode);

        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            var savedCode = await db.SavedCodes.FirstAsync();
            Assert.IsTrue(savedCode.IsPublic);
        }

        // Delete code
        var deleteResponse = await PostForm($"/Codes/Delete/{id}", new Dictionary<string, string>());
        AssertRedirect(deleteResponse, "/Codes/Index");

        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            var savedCodeExists = await db.SavedCodes.AnyAsync(c => c.Id == id);
            Assert.IsFalse(savedCodeExists);
        }
    }

    [TestMethod]
    public async Task TestOverwriteAndSaveResults()
    {
        await RegisterAndLoginAsync();

        // 1. Initial save
        var saveResponse = await PostForm("/Codes/Save", new Dictionary<string, string>
        {
            { "title", "Initial Title" },
            { "code", "initial code" },
            { "language", "cpp" },
            { "isPublic", "true" }
        });
        Assert.AreEqual(HttpStatusCode.OK, saveResponse.StatusCode);

        int id;
        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            id = (await db.SavedCodes.FirstAsync(c => c.Title == "Initial Title")).Id;
        }

        // 2. Overwrite with results
        var overwriteResponse = await PostForm("/Codes/Save", new Dictionary<string, string>
        {
            { "id", id.ToString() },
            { "title", "Updated Title" },
            { "code", "updated code" },
            { "language", "cpp" },
            { "isPublic", "true" },
            { "saveResult", "true" },
            { "result", "fake output" },
            { "error", "fake error" },
            { "resultCode", "0" }
        });
        Assert.AreEqual(HttpStatusCode.OK, overwriteResponse.StatusCode);

        // Verify in DB
        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            var savedCode = await db.SavedCodes.FirstOrDefaultAsync(c => c.Id == id);
            Assert.IsNotNull(savedCode);
            Assert.AreEqual("Updated Title", savedCode.Title);
            Assert.AreEqual("updated code", savedCode.Code);
            Assert.AreEqual("fake output", savedCode.Result);
            Assert.AreEqual("fake error", savedCode.Error);
            Assert.AreEqual(0, savedCode.ResultCode);
        }

        // 3. Save As (new record)
        var saveAsResponse = await PostForm("/Codes/Save", new Dictionary<string, string>
        {
            { "title", "New Title" },
            { "code", "new code" },
            { "language", "cpp" },
            { "isPublic", "true" }
        });
        Assert.AreEqual(HttpStatusCode.OK, saveAsResponse.StatusCode);

        using (var scope = Server!.Services.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<TemplateDbContext>();
            var initialCount = await db.SavedCodes.CountAsync(c => c.Id == id);
            var newCount = await db.SavedCodes.CountAsync(c => c.Title == "New Title");
            Assert.AreEqual(1, initialCount);
            Assert.AreEqual(1, newCount);
            
            var newCode = await db.SavedCodes.FirstAsync(c => c.Title == "New Title");
            Assert.AreNotEqual(id, newCode.Id);
            Assert.IsNull(newCode.Result); // Default is false for saveResult
        }
    }
}
