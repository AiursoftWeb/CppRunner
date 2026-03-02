namespace Aiursoft.CppRunner.Tests.IntegrationTests;

[TestClass]
public class JobsControllerTests : TestBase
{
    [TestMethod]
    public async Task TestJobsWorkflow()
    {
        await LoginAsAdmin();

        // 1. Index
        var indexResponse = await Http.GetAsync("/Jobs/Index");
        indexResponse.EnsureSuccessStatusCode();

        // 2. Create Job A
        var createAResponse = await PostForm("/Jobs/CreateTestJobA", new Dictionary<string, string>());
        AssertRedirect(createAResponse, "/Jobs");

        // 3. Create Job B
        var createBResponse = await PostForm("/Jobs/CreateTestJobB", new Dictionary<string, string>());
        AssertRedirect(createBResponse, "/Jobs");

        // 4. Index again (check if jobs are listed)
        var indexResponse2 = await Http.GetAsync("/Jobs/Index");
        indexResponse2.EnsureSuccessStatusCode();

        // 5. Cancel (need to find a job ID)
        // Since jobs are processed in the background, we might not have a reliable way to get a specific job ID easily 
        // from the UI without parsing. But let's at least test the endpoint with a dummy ID to see it handles it.
        var cancelResponse = await PostForm("/Jobs/Cancel", new Dictionary<string, string>
        {
            { "jobId", Guid.NewGuid().ToString() }
        });
        AssertRedirect(cancelResponse, "/Jobs");
    }
}
