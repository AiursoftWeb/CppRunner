using Aiursoft.UiStack.Layout;

namespace Aiursoft.CppRunner.Models.HomeViewModels;

public class LangInfo
{
    public string LangName { get; set; } = string.Empty;
    public string LangDisplayName { get; set; } = string.Empty;
    public string LangExtension { get; set; } = string.Empty;
    public string DefaultCode { get; set; } = string.Empty;
}

public class IndexViewModel : UiStackLayoutViewModel
{
    public IndexViewModel()
    {
        PageTitle = "Code Runner";
    }

    public List<LangInfo> Langs { get; set; } = [];

    public string? PreLoadedCode { get; set; }
    public string? PreLoadedLang { get; set; }
    public string? PreLoadedTitle { get; set; }
}
