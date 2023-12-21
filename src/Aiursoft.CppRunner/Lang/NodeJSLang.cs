namespace Aiursoft.CppRunner.Lang;

public class NodeJsLang : ILang
{
    public string LangDisplayName { get; set; } = "Javascript (Node.js v21)";
    
    public string LangExtension { get; set; } = "js";

    public string LangName { get; set; } = "javascript";

    public string DefaultCode { get; set; } = 
"""
function fibonacci() {
  let current = 1, next = 1;
  return function() {
      const temp = current;
      current = next;
      next = temp + current;
      return temp;
  };
}

const fib = fibonacci();
for (let i = 0; i < 20; i++) {
  console.log(fib());
}

""";

    public string EntryFileName { get; set; } = "main.js";
    public string DockerImage { get; set; } = "node:21-alpine";
    public string RunCommand { get; set; } = "node /app/main.js";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class TypeScriptLang : ILang
{
    public string LangDisplayName { get; set; } = "TypeScript (Node.js v21)";
    
    public string LangExtension { get; set; } = "ts";

    public string LangName { get; set; } = "typescript";

    public string DefaultCode { get; set; } = 
"""
function fibonacci(): () => number {
  let current = 1, next = 1;
  return function(): number {
      const temp = current;
      current = next;
      next = temp + current;
      return temp;
  };
}

const fib = fibonacci();
for (let i = 0; i < 20; i++) {
  console.log(fib());
}
""";

    public string EntryFileName { get; set; } = "main.ts";
    public string DockerImage { get; set; } = "node:21-alpine";
    public string RunCommand { get; set; } = "tsc /app/main.ts && node /app/main.js";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class RubyLang : ILang
{
    public string LangDisplayName { get; set; } = "Ruby (MRI 3.1)";

    public string LangExtension { get; set; } = "rb";

    public string LangName { get; set; } = "ruby";

    public string DefaultCode { get; set; } =
"""
def fibonacci
  Enumerator.new do |y|
    current, next = 1, 1
    loop do
      y << current
      current, next = next, current + next
    end
  end
end
""";

    public string EntryFileName { get; set; } = "main.rb";

    public string DockerImage { get; set; } = "ruby:3.1-alpine";

    public string RunCommand { get; set; } = "ruby /app/main.rb";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class HaskellLang : ILang
{
    public string LangDisplayName { get; set; } = "Haskell (GHC 9.2)";

    public string LangExtension { get; set; } = "hs";

    public string LangName { get; set; } = "haskell";

    public string DefaultCode { get; set; } =
"""
fibonacci :: [Integer]
fibonacci = 1 : 1 : zipWith (+) fibonacci (tail fibonacci)
""";

    public string EntryFileName { get; set; } = "main.hs";

    public string DockerImage { get; set; } = "haskell:9.2-alpine";

    public string RunCommand { get; set; } = "runhaskell /app/main.hs";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class GoLang : ILang
{
    public string LangDisplayName { get; set; } = "Go (Go 1.17)";

    public string LangExtension { get; set; } = "go";

    public string LangName { get; set; } = "go";

    public string DefaultCode { get; set; } =
    """
package main

import "fmt"

func fibonacci() func() int {
    current, next := 1, 1
    return func() int {
        result := current
        current, next = next, current + next
        return result
    }
}

func main() {
    fib := fibonacci()
    for i := 0; i < 20; i++ {
        fmt.Println(fib())
    }
}
""";

    public string EntryFileName { get; set; } = "main.go";

    public string DockerImage { get; set; } = "golang:1.17-alpine";

    public string RunCommand { get; set; } = "go run /app/main.go";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class SwiftLang : ILang
{
    public string LangDisplayName { get; set; } = "Swift (Swift 5.5)";

    public string LangExtension { get; set; } = "swift";

    public string LangName { get; set; } = "swift";

    public string DefaultCode { get; set; } =
    """
func fibonacci() -> () -> Int {
    var current = 1, next = 1
    return {
        let result = current
        current = next
        next = current + result
        return result
    }
}

let fib = fibonacci()
for _ in 0..<20 {
    print(fib())
}
""";

    public string EntryFileName { get; set; } = "main.swift";

    public string DockerImage { get; set; } = "swift:5.5-alpine";

    public string RunCommand { get; set; } = "swift /app/main.swift";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class RustLang : ILang
{
    public string LangDisplayName { get; set; } = "Rust (Rust 1.55)";

    public string LangExtension { get; set; } = "rs";

    public string LangName { get; set; } = "rust";

    public string DefaultCode { get; set; } =
    """
fn fibonacci() -> impl Iterator<Item = u64> {
    let mut current = 1;
    let mut next = 1;
    std::iter::from_fn(move || {
        let result = current;
        current = next;
        next = current + result;
        Some(result)
    })
}

fn main() {
    for i in fibonacci().take(20) {
        println!("{}", i);
    }
}
""";

    public string EntryFileName { get; set; } = "main.rs";

    public string DockerImage { get; set; } = "rust:1.55-alpine";

    public string RunCommand { get; set; } = "rustc /app/main.rs && /app/main";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class JavaLang : ILang
{
    public string LangDisplayName { get; set; } = "Java (OpenJDK 17)";

    public string LangExtension { get; set; } = "java";

    public string LangName { get; set; } = "java";

    public string DefaultCode { get; set; } =
    """
import java.util.stream.Stream;

public class Main {
    public static void main(String[] args) {
        Stream.iterate(new int[]{1, 1}, i -> new int[]{i[1], i[0] + i[1]})
                .map(i -> i[0])
                .limit(20)
                .forEach(System.out::println);
    }
}
""";

    public string EntryFileName { get; set; } = "Main.java";

    public string DockerImage { get; set; } = "openjdk:17-alpine";

    public string RunCommand { get; set; } = "javac /app/Main.java && java -cp /app Main";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class KotlinLang : ILang
{
    public string LangDisplayName { get; set; } = "Kotlin (Kotlin 1.5)";

    public string LangExtension { get; set; } = "kt";

    public string LangName { get; set; } = "kotlin";

    public string DefaultCode { get; set; } =
    """
fun fibonacci(): Sequence<Int> {
    var current = 1
    var next = 1
    return generateSequence {
        val result = current
        current = next
        next = current + result
        result
    }
}

fun main() {
    fibonacci().take(20).forEach(::println)
}
""";

    public string EntryFileName { get; set; } = "Main.kt";

    public string DockerImage { get; set; } = "kotlin:1.5-alpine";

    public string RunCommand { get; set; } = "kotlinc /app/Main.kt -include-runtime -d /tmp/main.jar && java -jar /tmp/main.jar";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class PhpLang : ILang
{
    public string LangDisplayName { get; set; } = "PHP (PHP 8.0)";

    public string LangExtension { get; set; } = "php";

    public string LangName { get; set; } = "php";

    public string DefaultCode { get; set; } =
    """
<?php
function fibonacci() {
    $current = 1;
    $next = 1;
    while (true) {
        yield $current;
        $next = $current + ($current = $next);
    }
}

foreach (fibonacci() as $i) {
    echo $i, PHP_EOL;
}
""";

    public string EntryFileName { get; set; } = "main.php";

    public string DockerImage { get; set; } = "php:8.0-alpine";

    public string RunCommand { get; set; } = "php /app/main.php";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}

public class PerlLang : ILang
{
    public string LangDisplayName { get; set; } = "Perl (Perl 5.34)";

    public string LangExtension { get; set; } = "pl";

    public string LangName { get; set; } = "perl";

    public string DefaultCode { get; set; } =
    """

sub fibonacci {
    my ($current, $next) = (1, 1);
    return sub {
        my $result = $current;
        ($current, $next) = ($next, $current + $next);
        return $result;
    };
}

my $fib = fibonacci();
for (1..20) {
    print $fib->(), "\n";
}
""";

    public string EntryFileName { get; set; } = "main.pl";

    public string DockerImage { get; set; } = "perl:5.34-alpine";

    public string RunCommand { get; set; } = "perl /app/main.pl";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}