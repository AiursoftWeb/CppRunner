const supportedLanguages = [
  { langName: "c", langDisplayName: "C (gcc 9.5.0)", langExtension: "c" },
  {
    langName: "cpp",
    langDisplayName: "C++ (GNU G++, stdc++20)",
    langExtension: "cpp",
  },
  {
    langName: "csharp",
    langDisplayName: "C# (.NET 7.0)",
    langExtension: "cs",
  },
  {
    langName: "go",
    langDisplayName: "Go (Golang 1.21.5)",
    langExtension: "go",
  },
  {
    langName: "rust",
    langDisplayName: "Rust (1.74.1)",
    langExtension: "rs",
  },
  {
    langName: "javascript",
    langDisplayName: "Javascript (Node.js v21)",
    langExtension: "js",
  },
];

const defaultCode = new Map([
  [
    "c",
    `#include <stdio.h>

  typedef struct FibonacciGenerator {
      int current;
      int next;
  } FibonacciGenerator;
  
  FibonacciGenerator fibonacci() {
      FibonacciGenerator fg = {1, 1};
      return fg;
  }
  
  int next(FibonacciGenerator* fg) {
      int result = fg->current;
      fg->current = fg->next;
      fg->next = fg->current + result;
      return result;
  }
  
  int main() {
      FibonacciGenerator fg = fibonacci();
      for (int i = 0; i < 20; i++) {
          printf("%d\n", next(&fg));
      }
      return 0;
  }
  `,
  ],
  [
    "cpp",
    `#include <iostream>
  #include <functional>
  
  std::function<int()> fibonacci()
  {
      int current = 1, next = 1;
      return [=]() mutable {
          int result = current;
          current = next;
          next = current + result;
          return result;
      };
  }
  
  int main()
  {
      auto fib = fibonacci();
      for (int i = 0; i < 20; i++) {
          std::cout << fib() << std::endl;
      }
      return 0;
  }
  `,
  ],
  [
    "csharp",
    `using System;
  using System.Collections.Generic;
  using System.Linq;
  
  public class Program
  {
      private static IEnumerable<int> Fibonacci()
      {
          int current = 1, next = 1;
  
          while (true)
          {
              yield return current;
              next = current + (current = next);
          }
      }
  
      public static void Main()
      {
          foreach (var i in Fibonacci().Take(20))
          {
              Console.WriteLine(i);
          }
      }
  }
  `,
  ],
  [
    "go",
    `package main

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
  }`,
  ],
  [
    "rust",
    `fn fibonacci() -> impl Iterator<Item = u64> {
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
}`,
  ],
  [
    "javascript",
    `function fibonacci() {
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
  `,
  ],
]);

const runCodeSuccess = new Map([
  [
    "c",
    {
      resultCode: 0,
      output:
        "1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n89\n144\n233\n377\n610\n987\n1597\n2584\n4181\n6765\n",
      error: "",
    },
  ],
  [
    "cpp",
    {
      resultCode: 0,
      output:
        "1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n89\n144\n233\n377\n610\n987\n1597\n2584\n4181\n6765\n",
      error: "",
    },
  ],
  [
    "csharp",
    {
      resultCode: 1,
      output:
        "1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n89\n144\n233\n377\n610\n987\n1597\n2584\n4181\n6765\n",
      error: "this is some error message",
    },
  ],
  [
    "go",
    {
      resultCode: 0,
      output:
        "1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n89\n144\n233\n377\n610\n987\n1597\n2584\n4181\n6765\n",
      error: "",
    },
  ],
  [
    "rust",
    {
      resultCode: 0,
      output:
        "1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n89\n144\n233\n377\n610\n987\n1597\n2584\n4181\n6765\n",
      error: "",
    },
  ],
  [
    "javascript",
    {
      resultCode: 0,
      output:
        "1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n89\n144\n233\n377\n610\n987\n1597\n2584\n4181\n6765\n",
      error: "",
    },
  ],
]);

export { supportedLanguages, defaultCode, runCodeSuccess };
