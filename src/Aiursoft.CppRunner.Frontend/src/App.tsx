import { MutableRefObject, useEffect, useRef, useState } from 'react'
import CodeEditor from '@uiw/react-textarea-code-editor';
import { getSupportLanguages, getDefaultCode, runCode } from './models/language';
import { State as ViewState } from './view-status';
import type { OutputResult } from './models/language';
import './styles/tailwind.css';

const OUTPUT_RENDER_MAX_LENGTH = 1024 * 2;

function App() {

  let runCodeController: MutableRefObject<AbortController | null> = useRef(null);
  let codeRef = useRef('');

  const [data, setData] = useState({
    languages: [],
    lang: '',
    langExtension: '',
    running: false
  } as ViewState);

  const [result, setResult] = useState({
    output: '',
    error: '',
  } as OutputResult);

  useEffect(() => {
    const execute = async () => {
      const langs = await getSupportLanguages();
      const newState = {
        ...data,
        languages: langs
      } as ViewState;

      if (langs.length !== 0) {
        const defaultCode = await getDefaultCode(langs[0].langName!);
        codeRef.current = defaultCode;
        newState.lang = langs[0].langName!;
        newState.langExtension = langs[0].langExtension!;
      }
      setData(newState);
    };

    execute();
  }, []);

  const langSelectRef: MutableRefObject<HTMLSelectElement | null> = useRef(null);

  const getSelectedLanguage = (): { lang:string, langExtension: string } => {
    let selected = langSelectRef.current!.selectedOptions[0];
    let langExtension = selected.getAttribute('data-lang-extension') ?? '';
    let lang = selected?.getAttribute('data-lang') ?? '';
    return {
      lang,
      langExtension
    }
  }

  const handleOutputClear = () => {
    setResult({
      ...result,
      output: '',
    } as OutputResult)
  }

  const handleErrorClear = () => {
    setResult({
      ...result,
      error: '',
    } as OutputResult)
  }

  const handleSelectLang = async (_: HTMLSelectElement) => {
    let { lang, langExtension } = getSelectedLanguage();
    const defaultCode = await getDefaultCode(lang);
    codeRef.current = defaultCode;
    setData({
      ...data,
      lang: lang,
      langExtension: langExtension
    } as ViewState);
    setResult({ output: '', error: '' } as OutputResult);
  }

  const handleEditorChange = async (code: string) => {
    codeRef.current = code;
  }

  const handleRun = async () => {
    try {
      setData({ ...data, running: true } as ViewState);
      setResult({ output: '', error: '' } as OutputResult);

      let { lang } = getSelectedLanguage();
      const [promise, controller] = runCode(lang, codeRef.current);
      runCodeController.current = controller;
      const fetchResult = await promise;

      if (runCodeController.current.signal.aborted) {
        return;
      }

      setResult({
        output: fetchResult.output,
        error: fetchResult.error
      } as OutputResult)
    } catch (error) {
      alert(`Oh, Exception occured: \n${error}\n`);
      console.error(error);
    }
    finally {
      // console.log(codeRef.current);
      setData({ ...data, running: false } as ViewState);
    }
  }

  const handleCancel = async () => {
    runCodeController.current?.abort();
  }

  return (
    <div className='flex flex-col h-full sm:h-screen'>
      <div className='w-screen h-[10vh] center text-lg'>
        Code Runner Online
      </div>
      <div className='flex flex-col sm:flex-row gap-4 sm:h-[85vh] w-full px-2'>

        <div className='flex flex-col flex-grow max-w-full sm:max-w-[50vw]'>
          <div className='flex items-center my-4 space-x-4'>
            <span>Input:</span>
            <select className="bg-gray-900 overflow-hidden"
              ref={langSelectRef} onChange={async (event) => { await handleSelectLang(event.target) }}>
              {
                Array.from(data.languages).map((lang) => (
                  <option key={lang.langName} data-lang={lang.langName} data-lang-extension={lang.langExtension} value={lang.langName}>{lang.langDisplayName}</option>
                ))
              }
            </select>
            <span className='flex-grow flex flex-row-reverse'>
              <span>
                {
                  data.running ?
                    <button className='p-2 border rounded' onClick={async () => { await handleCancel() }}>Cancel</button> :
                    <button className='p-2 border rounded' onClick={async () => { await handleRun() }}>Run</button>
                }
              </span>
            </span>
          </div>
          <div className='max-h-[50vh] sm:max-h-full sm:h-full overflow-scroll'>
            <div className='max-w-full sm:max-w-[50vw] max-h-[50%] sm:max-h-full sm:h-full'>
              <CodeEditor
                value={codeRef.current}
                language={data.langExtension}
                placeholder={`Enter ${data.lang} code:`}
                onChange={(evn) => handleEditorChange(evn.target.value)}
                padding={15}
                minHeight={100}
                data-color-mode="dark"
                style={{
                  backgroundColor: "#161b22",
                  minHeight: '100%',
                  fontFamily: 'ui-monospace,SFMono-Regular,SF Mono,Consolas,Liberation Mono,Menlo,monospace',
                }}
              />
            </div>
          </div>

        </div>

        <div className='flex-grow flex flex-col max-w-full sm:max-w-[50vw] space-y-4'>
          <div className='flex flex-col h-1/2 flex-grow'>
            <div className='flex items-center my-4 pr-4'>
              <span>Output:</span>
              <span className='flex-grow flex flex-row-reverse'>
                <button type='button' className='p-2' onClick={() => { handleOutputClear() }}>Clear</button>
              </span>
            </div>
            <div className='relative p-2 min-h-1/2 sm:min-h-1/2  sm:h-5/6 rounded border border-gray-600 break-words overflow-scroll'>
              {data.running && <div className='absolute bottom-1/2 left-1/2 z-50 center text-lg'>Running...</div>}
              {
                result.output!.length < OUTPUT_RENDER_MAX_LENGTH ?
                  (<CodeEditor
                    value={result.output}
                    language={data.langExtension}
                    padding={15}
                    minHeight={100}
                    disabled={true}
                    data-color-mode="dark"
                    style={{
                      backgroundColor: "#202b3c",
                      minHeight: '100%',
                      fontFamily: 'ui-monospace,SFMono-Regular,SF Mono,Consolas,Liberation Mono,Menlo,monospace',
                    }}
                  />) : (
                    <textarea className='h-full w-full bg-[#202b3c] resize-none outline-none'
                      value={result.output} readOnly>
                    </textarea>
                  )
              }
            </div>
          </div>

          <div className='flex flex-col flex-grow h-1/5'>
            <div className='flex items-center h-[20%] pr-4'>
              <span>Error:</span>
              <span className='flex-grow flex flex-row-reverse'>
                <button type='button' className='p-2' onClick={() => { handleErrorClear() }}>Clear</button>
              </span>
            </div>
            <div className='relative h-[30vh] sm:h-4/5 p-2 rounded border border-gray-600 break-words overflow-scroll'>
              {data.running && <div className='absolute bottom-1/2 left-1/2 z-50 center text-lg'>Running...</div>}
              {
                result.error!.length < OUTPUT_RENDER_MAX_LENGTH ?
                  (<CodeEditor
                    value={result.error}
                    language={data.langExtension}
                    padding={15}
                    minHeight={100}
                    disabled={true}
                    data-color-mode="dark"
                    style={{
                      backgroundColor: "#202b3c",
                      minHeight: '100%',
                      fontFamily: 'ui-monospace,SFMono-Regular,SF Mono,Consolas,Liberation Mono,Menlo,monospace',
                    }}
                  />) : (
                    <textarea className='h-full w-full bg-[#202b3c] resize-none outline-none'
                      value={result.error} readOnly>
                    </textarea>
                  )
              }

            </div>
          </div>
        </div>
      </div>

      <footer className='center h-[5vh] space-x-4'>
        <a className='text-blue-500' href='https://gitlab.aiursoft.com/aiursoft/cpprunner' target='_blank'>Source Code</a>
      </footer>
    </div>

  )
}

export default App
