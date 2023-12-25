namespace Aiursoft.CppRunner.Models;

public class CodeResult
{
    public int ResultCode { get; set; }
    
    public string Output { get; set; } = string.Empty;
    
    public string Error { get; set; } = string.Empty;
}