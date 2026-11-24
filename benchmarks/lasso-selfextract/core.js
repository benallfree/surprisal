/*
Shared predictor for the encoder AND the decoder. This exact text is shipped inside the
self-extracting page and also run by pack.mjs, so both sides make identical predictions.

Z(n, f) runs the model over n bytes, one bit at a time (MSB first). For each bit it calls
f(p), where p is the model's probability (1..4095 out of 4096) that the bit is 1, and f
returns the actual bit: the encoder knows it and arithmetic-codes it; the decoder recovers
it from the code stream. Z returns the bytes it saw.

Model: 7 context models (orders 0,1,2,3,4,6 plus "current word"), each a hashed table of
adaptive bit probabilities, combined by a logistic mixer whose weights are picked by the
partial byte and learn online. No pretrained weights: it starts blank and learns the file
as it goes, exactly the "fine-tune on the fly" idea, so the decoder ships zero parameters.

Only + - * / Math.imul Math.floor and bit ops are used (no Math.exp/log): those are exact in
every JS engine, so a file packed in Node decodes bit-identically in any browser.
*/
Z=(n,f)=>{let w,G=R5,N=G.length+1,B=20,O=new Uint8Array(n),P=new Uint16Array(N<<B).fill(32768),C=new Uint8Array(N<<B),W=new Float64Array(N<<8).fill(R1),S=[],Q=[],H=[],I=[],A=[],h=0,i=0,j,k,c,p,d,e,t,y,m;for(d=-2047;d<2048;d++){for(e=t=k=1;k<60;)e+=t*=-d/256/k++;j=Math.round(4096/(1+e));Q[d]=j<1?1:j>4095?4095:j}for(d=-2047,k=0;d<2048;d++)for(;k<=Q[d];)S[k++]=d;for(;k<4096;)S[k++]=2047;for(;i<n;i++){for(m=0;m<N-1;m++)for(H[m]=m,k=1;k<=G[m];)H[m]=Math.imul(H[m],0x2F0B4AC7)^O[i-k++];c=O[i-1];h=c>47&&c<58||c>64&&c<91||c>96&&c<123||c==95?Math.imul(h+c+1,0x6F4F2A35):0;H[N-1]=h+N;for(c=1;c<256;){for(w=R4,d=m=0;m<N;m++)I[m]=(Math.imul(H[m]+c*40503,0x2C9277B5)>>>32-B)+(m<<B),d+=W[w*N+m]*(A[m]=S[P[I[m]]>>4]);d=Math.floor(d/65536);p=Q[d<-2047?-2047:d>2047?2047:d];y=f(p);e=((y<<12)-p)*R2;for(m=0;m<N;m++)W[w*N+m]+=A[m]*e/1024,j=I[m],P[j]+=(y*65535-P[j])/(C[j]+1.5)|0,C[j]<R3&&C[j]++;c=c*2+y}O[i]=c&255}return O}
