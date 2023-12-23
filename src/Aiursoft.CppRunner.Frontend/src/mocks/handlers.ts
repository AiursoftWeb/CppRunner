import { HttpResponse, http, delay } from "msw";
import { supportedLanguages, defaultCode, runCodeSuccess } from "./fake-data";
import type { OutputResult } from "../models/language";

export const handlers = [
  http.get("/langs", () => {
    return HttpResponse.json(supportedLanguages);
  }),

  http.get("/langs/:lang/default", ({ params }) => {
    const { lang } = params;
    const code = defaultCode.get(lang as string);

    return HttpResponse.text(code);
  }),

  http.post("/runner/run", async ({ request }) => {
    const lang = new URLSearchParams(request.url.split("?")[1]).get("lang")!;

    await delay(2000);
    const result = runCodeSuccess.get(lang)!;
    return getFakeResponse(lang, result);
  }),
];

function getFakeResponse(lang: string, result: OutputResult) {
  switch (lang) {
    case "cpp": {
      return new HttpResponse(null, {
        status: 429,
        statusText: "Too Many Requests",
      });
    }
    default:
      return HttpResponse.json(result);
  }
}
