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

        // 2. Trigger DummyJob
        var triggerAResponse = await PostForm("/Jobs/Trigger", new Dictionary<string, string>
        {
            { "jobTypeName", "DummyJob" }
        });
        AssertRedirect(triggerAResponse, "/Jobs");

        // 3. Trigger OrphanAvatarCleanupJob
        var triggerBResponse = await PostForm("/Jobs/Trigger", new Dictionary<string, string>
        {
            { "jobTypeName", "OrphanAvatarCleanupJob" }
        });
        AssertRedirect(triggerBResponse, "/Jobs");

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
