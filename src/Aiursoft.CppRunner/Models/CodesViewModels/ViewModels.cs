using Aiursoft.CppRunner.Entities;
using Aiursoft.UiStack.Layout;

namespace Aiursoft.CppRunner.Models.CodesViewModels;

public class IndexViewModel : UiStackLayoutViewModel
{
    public IndexViewModel()
    {
        PageTitle = "My Codes";
    }

    public required IEnumerable<SavedCode> MyCodes { get; set; }
}

public class PublicViewModel : UiStackLayoutViewModel
{
    public PublicViewModel()
    {
        PageTitle = "Public Codes";
    }

    public required IEnumerable<SavedCode> PublicCodes { get; set; }
}
