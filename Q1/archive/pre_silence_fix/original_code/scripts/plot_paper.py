"""从可追溯 Q1 输出生成论文插图；所有绘图说明集中在 README，图内不放脚注。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ft2font import FT2Font
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from artifact_contract import file_hash
from pipeline import decode_media
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE, PINK, INK, GRID = '#6B91AF', '#C9919A', '#354B5B', '#E5EBEF'


def configure_font(explicit):
    candidates = [explicit] if explicit else []
    candidates += [Path(f.fname) for f in font_manager.fontManager.ttflist
                   if any(n in f.name for n in ['Noto Sans CJK', 'WenQuanYi', 'Arial Unicode', 'Heiti', 'Songti'])]
    required = set(map(ord, '原始视频文本声学特征视觉对齐时间秒样本词覆盖率输入输出模型'))
    for path in candidates:
        if path and Path(path).is_file() and required <= set(FT2Font(str(path)).get_charmap()):
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams.update({'font.family': name, 'font.size': 10, 'axes.unicode_minus': False,
                                 'ps.fonttype': 3, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
                                 'axes.spines.right': False, 'axes.labelcolor': INK,
                                 'text.color': INK, 'savefig.facecolor': 'white'})
            return str(path)
    raise RuntimeError('未找到具备中文字形的字体，请通过 --font 指定 CJK 字体文件')


def save(fig, name, dest):
    dest.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig,json_out=dest/f'{name}.alignment.json',strict=True)
    # Two exports from the same Matplotlib figure; no ImageGen or external image synthesis.
    fig.savefig(dest/f'{name}.png', dpi=600, bbox_inches='tight', pad_inches=.07)
    fig.savefig(dest/f'{name}.eps', dpi=600, bbox_inches='tight', pad_inches=.07)
    fig.savefig(dest/f'{name}.pdf', bbox_inches='tight', pad_inches=.07)
    fig.savefig(dest/f'{name}.svg', bbox_inches='tight', pad_inches=.07)
    plt.close(fig)


def grid(ax):
    ax.grid(axis='y', color=GRID, lw=.5)
    ax.set_axisbelow(True)


def workflow(dest):
    labels = ['视频与转写', '时间戳解码', '三模态提取', '词级强制对齐', '词区间池化', '子词映射', '掩码与质检']
    widths = [max(1.35, .22*len(label)+.3) for label in labels]
    gap = .28
    total = sum(widths)+(len(widths)-1)*gap
    fig, ax = plt.subplots(figsize=(total, 2.45))
    ax.set(xlim=(-.08, total+.08), ylim=(-1.13, .95)); ax.axis('off')
    positions=[]; x=0
    for i, (label, w) in enumerate(zip(labels,widths)):
        positions.append((x,w))
        ax.add_patch(FancyBboxPatch((x, .22), w, .5, boxstyle='round,pad=0.02,rounding_size=.06',
                                  fc='#E4EDF4' if i<3 else '#F2E3E7', ec='#8296A5', lw=.7))
        ax.text(x+w/2,.47,label,ha='center',va='center',fontsize=10)
        if i:
            prev,pw=positions[i-1]
            ax.add_patch(FancyArrowPatch((prev+pw+.04,.47),(x-.04,.47),arrowstyle='-|>',mutation_scale=9,lw=.7,color=INK))
        x += w+gap
    # Model names identify the contents of the extraction stage; they are not invented processing steps.
    cx=positions[2][0]+positions[2][1]/2
    branch_labels=['BERT 语义','F0 / MFCC','YuNet 几何']
    branch_w=1.55; centers=[cx-1.85,cx,cx+1.85]
    ax.plot([cx,cx], [.19,-.15],color=INK,lw=.7)
    ax.plot([centers[0],centers[-1]],[-.15,-.15],color=INK,lw=.7)
    for center,label in zip(centers,branch_labels):
        ax.add_patch(FancyBboxPatch((center-branch_w/2,-.88),branch_w,.46,
            boxstyle='round,pad=0.02,rounding_size=.06',fc='#F1F5F8',ec='#8296A5',lw=.7))
        ax.add_patch(FancyArrowPatch((center,-.15),(center,-.37),arrowstyle='-|>',mutation_scale=9,lw=.7,color=INK))
        ax.text(center,-.65,label,ha='center',va='center',fontsize=10)
    save(fig,'00_q1_workflow',dest)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'outputs')
    p.add_argument('--figures',type=Path,default=ROOT/'figures/paper')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--font',type=Path)
    args=p.parse_args(); font=configure_font(args.font)
    records=sorted([json.loads(x.read_text()) for x in (args.out/'records').glob('*.json')],key=lambda r:r['sample_id'])
    if len(records)!=100: raise ValueError('需要完整的 100 条结果')
    workflow(args.figures)
    durations=np.array([r['media']['container_duration_s'] for r in records])
    fig,axs=plt.subplots(1,2,figsize=(8,2.65),layout='constrained')
    axs[0].hist(durations,bins=np.linspace(0,np.ceil(durations.max()/3)*3,12),color=BLUE,edgecolor='white')
    axs[0].set(xlabel='视频时长 / s',ylabel='样本数')
    axs[1].plot(np.arange(1,101),np.sort(durations),color=BLUE,lw=1.2)
    axs[1].set(xlabel='按时长排序的样本序号',ylabel='视频时长 / s',ylim=(0,None))
    for ax in axs: grid(ax)
    save(fig,'01_input_duration',args.figures)
    median=np.median(durations)
    eligible=[r for r in records if r['face_detection_rate']>=.9 and r['word_coverage']>=.98]
    rep=min(eligible or records,key=lambda r:abs(r['media']['container_duration_s']-median));sid=rep['sample_id']
    doc=json.loads((args.out/'alignment'/f'{sid}.json').read_text())
    with np.load(args.out/'features'/f'{sid}.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
    words=[w for w in doc['words'] if w['aligned']]; mid=max(0,len(words)//2-3);shown=words[mid:mid+6]
    start=max(doc['media']['audio_start_s'],shown[0]['start']-.12)
    end=min(doc['media']['audio_end_s'],shown[-1]['end']+.12)
    vid,clip=sid.rsplit('__',1)
    wave,_,a0,_,_,_,_=decode_media(args.data_root/vid/f'{clip}.mp4')
    fig,axs=plt.subplots(3,1,figsize=(8,4.35),sharex=True,gridspec_kw={'height_ratios':[1,1,1.3]})
    fig.subplots_adjust(left=.10,right=.88,bottom=.13,top=.97,hspace=.24)
    for j,w in enumerate(shown):
        y=.25 if j%2==0 else .72
        axs[0].plot([w['start'],w['end']],[y,y],lw=4,color=PINK if w['score']<.2 else BLUE,solid_capstyle='butt')
        axs[0].text((w['start']+w['end'])/2,y+.10,w['text'],ha='center',va='bottom',fontsize=9)
    axs[0].set(ylim=(0,1.16),yticks=[],ylabel='自动词区间')
    wt=np.arange(len(wave))/16000+a0;mask=(wt>=start)&(wt<=end)
    axs[1].plot(wt[mask],wave[mask],lw=.45,color=BLUE);axs[1].set(ylabel='波形幅值')
    mask=(d['audio_times']>=start)&(d['audio_times']<=end);times=d['audio_times'][mask]
    mfcc=d['raw_audio'][mask,2:].T;vmax=np.percentile(abs(mfcc),98)
    edges=np.r_[times-.025,times[-1]+.025]
    im=axs[2].pcolormesh(edges,np.arange(26),mfcc,cmap=LinearSegmentedColormap.from_list('muted_blue_pink',['#6B91AF','#FAFAFA','#C9919A']),vmin=-vmax,vmax=vmax,shading='flat',rasterized=False)
    axs[2].set(xlabel='相对视频起点时间 / s',ylabel='MFCC 系数序号',xlim=(start,end))
    pos=axs[2].get_position()
    cax=fig.add_axes([.90,pos.y0,.015,pos.height],label='<colorbar>')
    bar=fig.colorbar(im,cax=cax,label='系数值')
    bar.solids.set_rasterized(False)
    save(fig,'02a_word_audio_alignment',args.figures)
    fig,axs=plt.subplots(2,1,figsize=(8,3.65),sharex=True,layout='constrained')
    pos=[(w['start']+w['end'])/2 for w in shown]
    norms=[np.linalg.norm(d['text'][w['token_indices']],axis=1).mean() for w in shown]
    axs[0].plot(pos,norms,'o-',color=BLUE,lw=1,ms=4);axs[0].set(ylabel='BERT 特征 L2 范数')
    v=d['raw_vision'];eye=np.linalg.norm(v[:,5:7]-v[:,7:9],axis=1);mouth=np.linalg.norm(v[:,11:13]-v[:,13:15],axis=1)
    ratio=np.divide(mouth,eye,out=np.full_like(mouth,np.nan),where=d['raw_vision_valid']&(eye>1e-6))
    visual_window=(d['vision_times']>=start)&(d['vision_times']<=end)
    axs[1].plot(d['vision_times'][visual_window],ratio[visual_window],'.-',color=PINK,lw=1,ms=3)
    axs[1].set(xlabel='相对视频起点时间 / s',ylabel='嘴宽 / 眼间距',xlim=(start,end))
    for ax in axs: grid(ax)
    save(fig,'02b_text_visual_features',args.figures)
    fig,axs=plt.subplots(2,1,figsize=(8,3.6),sharex=True,layout='constrained');x=np.arange(1,101)
    axs[0].plot(x,[100*r['word_coverage'] for r in records],'o',color=BLUE,ms=2.8)
    axs[0].set(ylabel='自动词覆盖率 / %',ylim=(-3,103))
    axs[1].bar(x,[100*r['face_detection_rate'] for r in records],color=BLUE,width=.8)
    axs[1].set(xlabel='按样本 ID 排序的序号',ylabel='人脸检出率 / %',ylim=(0,105))
    for ax in axs: grid(ax); ax.set_xlim(.3,100.7)
    save(fig,'03_feature_validity',args.figures)
    # Exact lookups remain data tables instead of being rasterized into a figure.
    table=pd.DataFrame([{'原词':w['text'],'开始时间_s':w['start'],'结束时间_s':w['end'],
        'WordPiece索引':','.join(map(str,w['token_indices'])),
        '声学帧索引':','.join(map(str,w['audio_indices'])),
        '视觉帧索引':','.join(map(str,w['vision_indices'])),
        '原视频帧索引':','.join(map(str,d['video_frame_indices'][w['vision_indices']].tolist())),
        'CTC发射分数':w['score'],'声学回退':w.get('audio_fallback',False),'视觉回退':w.get('vision_fallback',False)} for w in shown])
    table.to_csv(args.out/'04_token_feature_map.csv',index=False)
    table.to_excel(args.out/'04_token_feature_map.xlsx',index=False)
    # Display detection observations from all zero-detection clips, without synthetic faces.
    failures=[r for r in records if r['face_detection_rate']==0]
    if failures:
        fig,ax=plt.subplots(figsize=(8,max(2, .45*len(failures)+1)),layout='constrained')
        for i,r in enumerate(failures):
            with np.load(args.out/'features'/f'{r["sample_id"]}.npz',allow_pickle=False) as z:
                ax.plot(z['vision_times'],np.full(len(z['vision_times']),i),'|',color=PINK,ms=10,mew=.7)
        ax.set(yticks=range(len(failures)),yticklabels=[r['sample_id'] for r in failures],xlabel='相对视频起点时间 / s',ylabel='零检出样本')
        save(fig,'05_zero_detection_samples',args.figures)
    fig,axs=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
    lengths=[r['token_count'] for r in records]
    axs[0].hist(lengths,bins=np.arange(0,max(lengths)+10,10),color=BLUE,edgecolor='white')
    axs[0].set(xlabel='WordPiece 序列长度',ylabel='样本数')
    axs[1].scatter(durations,[100*r['face_detection_rate'] for r in records],s=17,color=PINK,edgecolors=INK,linewidths=.3)
    axs[1].set(xlabel='视频时长 / s',ylabel='人脸检出率 / %',ylim=(-3,103));grid(axs[1])
    save(fig,'06_length_and_detection',args.figures)
    receipt={'generator':str(Path(__file__).name),'script_sha256':file_hash(__file__), 'font':font,
        'representative_sample':sid,'window_s':[start,end],'words':[w['text'] for w in shown],
        'selection':'closest to median duration among face_detection_rate>=.90 and word_coverage>=.98; no labels used',
        'source_environment_sha256':file_hash(args.out/'environment.json'),
        'feature_hashes':{r['sample_id']:file_hash(args.out/'features'/f'{r["sample_id"]}.npz') for r in records},
        'image_generation':'Matplotlib only; no synthetic data, no generated human images',
        'human_boundary_validation':'pending','zero_detection_ids':[r['sample_id'] for r in failures]}
    (args.figures/'sources.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(f'Figures saved to {args.figures}; native table saved to {args.out}')


if __name__=='__main__': main()
