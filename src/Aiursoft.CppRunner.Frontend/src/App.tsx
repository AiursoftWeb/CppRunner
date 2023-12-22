import { MutableRefObject, useEffect, useRef, useState } from 'react'
import CodeEditor from '@uiw/react-textarea-code-editor';
import { getSupportLanguages, getDefaultCode, runCode } from './models/language';
import { State as ViewState } from './view-status';
import type { OutputResult } from './models/language';
import './styles/tailwind.css';

function App() {

  const [data, setData] = useState({
    languages: [],
    code: '',
    lang: '',
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
        newState.code = defaultCode;
        newState.lang = langs[0].langName!;
      }
      setData(newState);
    };

    execute();
  }, []);

  const langSelectRef: MutableRefObject<HTMLSelectElement | null> = useRef(null);

  const handleOutputClear = () => {
    setResult({
      ...result,
      output: '',
    } as OutputResult)
  }

  const handleErrorClear = () => {
    setResult({
      ...result,
      errer: '',
    } as OutputResult)
  }

  const handleSelectLang = async (lang: string) => {
    const defaultCode = await getDefaultCode(lang);
    setData({
      ...data,
      code: defaultCode,
      lang: lang
    } as ViewState);
    setResult({ output: '', error: '' } as OutputResult);
  }

  const handleEditorChange = async (code: string) => {
    setData({ ...data, code } as ViewState);
  }

  const handleRun = async () => {
    try {
      setData({ ...data, running: true } as ViewState);
      setResult({ output: '', error: '' } as OutputResult);
      const lang = langSelectRef.current!.value;
      const result: OutputResult = await runCode(lang, data.code);
      setResult({
        output: result.output,
        error: result.error
      } as OutputResult)
    } catch (error) {
      alert(`Oh, Exception occured: \n${error}`);
      console.error(error);
    }
    finally {
      setData({ ...data, running: false } as ViewState);
    }
  }

  return (
    <div className='flex flex-col h-screen'>
      <div className='w-screen h-[10vh] center text-lg'>
        Anduin's code background
      </div>
      <div className='flex flex-row gap-4 h-[85vh] w-full px-2'>
        <div className='flex flex-col flex-grow max-w-[50vw]'>
          <div className='flex items-center my-4 space-x-4'>
            <span>Input:</span>
            <select className="bg-gray-900"
              ref={langSelectRef} onChange={async (event) => { await handleSelectLang(event.target.value) }}>
              {
                Array.from(data.languages).map((lang) => (
                  <option key={lang.langName} value={lang.langName}>{lang.langDisplayName}</option>
                ))
              }
            </select>
            <span className='flex-grow flex flex-row-reverse'>
              <span><button className='p-2' disabled={data.running} onClick={async () => { await handleRun() }}>Run</button></span>
            </span>
          </div>
          <div className='h-full overflow-scroll'>
            <div className='max-w-[50vw] h-full'>
              <CodeEditor
                value={data.code}
                language={data.lang}
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

        <div className='flex-grow flex flex-col max-w-[50vw] space-y-4'>
          <div className='flex flex-col h-1/2 flex-grow'>
            <div className='flex items-center my-4 pr-4'>
              <span>Output:</span>
              <span className='flex-grow flex flex-row-reverse'>
                <button type='button' className='p-2' onClick={() => { handleOutputClear() }}>Clear</button>
              </span>
            </div>
            <div className='relative p-2 h-5/6 rounded border border-gray-600 break-words overflow-scroll'>
              {data.running && <div className='absolute bottom-1/2 left-1/2 z-50 center text-lg'>Running...</div>}
              <CodeEditor
                value={result.output}
                language={data.lang}
                padding={15}
                minHeight={100}
                disabled={true}
                data-color-mode="dark"
                style={{
                  backgroundColor: "#202b3c",
                  minHeight: '100%',
                  fontFamily: 'ui-monospace,SFMono-Regular,SF Mono,Consolas,Liberation Mono,Menlo,monospace',
                }}
              />
            </div>
          </div>

          <div className='flex flex-col flex-grow h-1/5'>
            <div className='flex items-center h-[20%] pr-4'>
              <span>Error:</span>
              <span className='flex-grow flex flex-row-reverse'>
                <button type='button' className='p-2' onClick={() => { handleErrorClear() }}>Clear</button>
              </span>
            </div>
            <div className='relative h-[80%] p-2 rounded border border-gray-600 break-words overflow-scroll'>
              {data.running && <div className='absolute bottom-1/2 left-1/2 z-50 center text-lg'>Running...</div>}
              <CodeEditor
                value={result.error}
                language={data.lang}
                padding={15}
                minHeight={100}
                disabled={true}
                data-color-mode="dark"
                style={{
                  backgroundColor: "#202b3c",
                  minHeight: '100%',
                  fontFamily: 'ui-monospace,SFMono-Regular,SF Mono,Consolas,Liberation Mono,Menlo,monospace',
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <footer className='center h-[5vh] space-x-4'>
        <span>Page Created by Dvorak</span>
        <a className='text-blue-500' href='https://gitlab.aiursoft.cn/aiursoft/cpprunner' target='_blank'>CppRunner</a>
      </footer>
    </div>

  )
}

export default App
