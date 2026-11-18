"use strict";
window.loadResearchData=async function(url){
  const response=await fetch(url);
  if(!response.ok)throw Error('研究数据下载失败，请刷新重试。');
  const bytes=new Uint8Array(await response.arrayBuffer());
  const text=bytes[0]===31&&bytes[1]===139
    ?await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text()
    :new TextDecoder().decode(bytes);
  return JSON.parse(text);
};
window.showResearchDataError=function(error){
  const notice=document.createElement('p');notice.className='research-notice';notice.setAttribute('role','alert');
  notice.textContent='研究数据暂时无法加载，请使用支持 gzip 解压的现代浏览器并刷新重试。';
  document.querySelector('main')?.prepend(notice);console.error(error);
};
