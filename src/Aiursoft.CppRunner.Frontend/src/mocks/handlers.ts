import { HttpResponse, http, delay } from "msw";
import { supportedLanguages, defaultCode, runCodeSuccess } from "./fake-data";

export const handlers = [
  http.get("/langs", () => {
    return HttpResponse.json(supportedLanguages);
  }),

  http.get("/langs/:lang/default", ({ params }) => {
    const { lang } = params;
    const code = defaultCode.get(lang as string);

    return HttpResponse.text(code);
  }),

  http.post("/runner/run?lang=:lang", async ({ request }) => {
    const lang = new URLSearchParams(request.url.split("?")[1]).get("lang");

    await delay(2000);
    const result = runCodeSuccess.get(lang as string);
    return HttpResponse.json(result);
  }),
];
