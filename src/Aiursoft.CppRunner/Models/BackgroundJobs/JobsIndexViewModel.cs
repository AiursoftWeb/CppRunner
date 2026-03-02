using Aiursoft.UiStack.Layout;

namespace Aiursoft.CppRunner.Models.BackgroundJobs;

public class JobsIndexViewModel : UiStackLayoutViewModel
{
    public IEnumerable<JobInfo> AllRecentJobs { get; init; } = [];
}
