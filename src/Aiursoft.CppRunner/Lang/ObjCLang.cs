namespace Aiursoft.CppRunner.Lang;

public class ObjCLang : ILang
{
    public string LangDisplayName { get; set; } = "Objective-C (Clang 13)";

    public string LangExtension { get; set; } = "m";

    public string LangName { get; set; } = "objective-c";

    public string DefaultCode { get; set; } =
        """
        #import <Foundation/Foundation.h>

        int main() {
            @autoreleasepool {
                int current = 1, next = 1;
                for (int i = 0; i < 20; i++) {
                    NSLog(@"%d", current);
                    next = current + (current = next);
                }
            }
        }
        """;

    public string EntryFileName { get; set; } = "main.m";

    public string DockerImage { get; set; } = "clang:13-alpine";

    public string RunCommand { get; set; } = "clang /app/main.m -o /tmp/main -framework Foundation && /tmp/main";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}