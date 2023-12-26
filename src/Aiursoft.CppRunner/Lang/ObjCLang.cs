namespace Aiursoft.CppRunner.Lang;

public class ObjCLang : ILang
{
    public string LangDisplayName => "Objective-C (Clang 13)";

    public string LangExtension => "m";

    public string LangName => "objective-c";

    public string DefaultCode =>
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

    public string EntryFileName => "main.m";

    public string DockerImage => "clang:13-alpine";

    public string RunCommand => "clang /app/main.m -o /tmp/main -framework Foundation && /tmp/main";

    public Dictionary<string, string> OtherFiles => new();
}