"""Intuitive, data-rich schematic figures for the FLINT paper (hand-drawn diagrams).

Minimalist, spacious flow -- NO heavy stage boxes: steps are separated by subtle dotted
dividers and connected by bold straight arrows. The overview uses a TWO-ROW layout for
breathing room. Threaded through ONE clean, universally understood, MULTI-ROW worked
example (a Country / Capital / Population table -> Wikidata).

  task.pdf     : WHAT the problem is  -- the table grounded in a KG fragment; CTA = a class
                 per entity column (green), CPA = a property per column pair (amber).
  overview.pdf : HOW FLINT solves it -- rich 5-step visual flow over two rows
                 (link+candidates -> feature matrix -> boosted-tree ranker  ==>
                  argmax + arborescence decode -> output semantic graph).

Colour = high-contrast, colourblind-aware, by role:
  BLUE entity | GREEN class/type (CTA) | AMBER property/relation (CPA) | VERMILLION FLINT core.
Wikidata IDs are REAL/verified: France Q142, Paris Q90 (distractors Paris,TX Q830149,
Paris Hilton Q47899), country Q6256, capital city Q5119; P36 capital, P1376 capital-of
(inverse), P1082 population. Scores/features are an illustrative-but-consistent example.

Run:  python3 scripts/make_schematic.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

FIG = Path(__file__).resolve().parent.parent / "paper/flint/figures"
FIG.mkdir(parents=True, exist_ok=True)

INK   = "#1C2B39"                         # deep navy ink (softer than pure black)
ENT   = "#2D7DD2"; ENT_BG="#E6F1FB"       # entity   - bright azure
CLS   = "#12A17C"; CLS_BG="#E2F7F0"; CLS_DK="#0B6B52"   # class/CTA - fresh teal
PROP  = "#F0A01E"; PROP_DK="#9A6300"      # property/CPA - marigold (clearly warm-gold)
CORE  = "#E8452F"; CORE_BG="#FCE7E2"      # FLINT core   - punchy coral-red
SLATE = "#5A6B7B"
MUTE  = "#98A6B0"; FAINT="#C7CDD4"; DARK="#1a1a1a"
STEEL = "#DDE9F4"; GRID="#1AA6C4"         # heatmap - cyan-teal (distinct from CTA green)

plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42,
                     "savefig.bbox":"tight","savefig.pad_inches":0.05})


# ---------- primitive helpers -----------------------------------------------
def rrect(ax,x,y,w,h,fc="white",ec=INK,lw=1.4,r=0.4,z=2,alpha=1.0,ls="-"):
    ax.add_patch(FancyBboxPatch((x+r,y+r),w-2*r,h-2*r,boxstyle=f"round,pad={r}",
                 fc=fc,ec=ec,lw=lw,zorder=z,alpha=alpha,linestyle=ls))

def rect(ax,x,y,w,h,fc="white",ec=INK,lw=1.3,z=2):
    ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw,zorder=z))

def arrow(ax,x1,y1,x2,y2,c=INK,lw=2.0,style="-|>",ms=13,ls="-",rad=None,z=6,sh=4):
    cs=f"arc3,rad={rad}" if rad is not None else None
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=ms,
                 color=c,lw=lw,linestyle=ls,zorder=z,shrinkA=sh,shrinkB=sh,connectionstyle=cs))

def dotv(ax,x,y1,y2,c=FAINT,lw=1.2):
    ax.plot([x,x],[y1,y2],ls=(0,(1,3)),color=c,lw=lw,zorder=1)

def doth(ax,x1,x2,y,c=FAINT,lw=1.2):
    ax.plot([x1,x2],[y,y],ls=(0,(1,3)),color=c,lw=lw,zorder=1)

def ell(ax,cx,cy,text,fc=ENT,w=15,h=7.2,tc="white",fs=6.4,ec=INK,lw=1.4,z=4):
    ax.add_patch(Ellipse((cx,cy),w,h,fc=fc,ec=ec,lw=lw,zorder=z))
    ax.text(cx,cy,text,ha="center",va="center",fontsize=fs,color=tc,weight="bold",zorder=z+1)

def clschip(ax,cx,cy,text,w=20,h=9,fs=7.0):
    rrect(ax,cx-w/2,cy-h/2,w,h,fc=CLS_BG,ec=CLS,lw=1.8,r=0.9,z=4)
    ax.text(cx,cy,text,ha="center",va="center",fontsize=fs,color=CLS_DK,weight="bold",zorder=6)

def steplabel(ax,cx,y,i,txt,c,fs=8.4):
    # centre the [number][label] group on cx (approx text width in data units)
    ppu=9.8*72/100.0                      # points per data unit at this figsize/xlim
    tw=len(txt)*0.52*fs/ppu               # estimated label width
    cd,gap=3.4,1.3; gw=cd+gap+tw; left=cx-gw/2; ccx=left+cd/2
    ax.add_patch(Ellipse((ccx,y),cd,cd,fc=c,ec="none",zorder=6))
    ax.text(ccx,y,str(i),ha="center",va="center",fontsize=6.8,weight="bold",color="white",zorder=7)
    ax.text(left+cd+gap,y,txt,ha="left",va="center",fontsize=fs,weight="bold",color=c,zorder=7)

def check(ax,cx,cy,s=1.0,c=CLS):
    ax.plot([cx-1.1*s,cx-0.2*s,cx+1.4*s],[cy,cy-1.0*s,cy+1.2*s],color=c,lw=1.8*s,solid_capstyle="round",zorder=7)

def table(ax,x,y,cw,ch,headers,rows,fs_h=7.2,fs_b=6.8,hi=None):
    xs=[x+j*cw for j in range(len(headers))]
    for j,hd in enumerate(headers):
        rect(ax,xs[j],y,cw,ch,fc=STEEL,ec=INK,lw=1.1,z=3)
        ax.text(xs[j]+cw/2,y+ch/2,hd,ha="center",va="center",fontsize=fs_h,weight="bold",color=INK,zorder=4)
    for i,rw in enumerate(rows):
        yy=y-(i+1)*ch
        for j,cell in enumerate(rw):
            fc="#FFF0C2" if hi==(i,j) else "white"
            rect(ax,xs[j],yy,cw,ch,fc=fc,ec=INK,lw=0.9,z=3)
            ax.text(xs[j]+cw/2,yy+ch/2,cell,ha="center",va="center",fontsize=fs_b,color=INK,zorder=4)
    return [xj+cw/2 for xj in xs]

def scorebars(ax,x0,ytop,items,wmax=17,bh=2.7,gap=2.1,fs=6.0):
    for i,(lab,val,hl) in enumerate(items):
        yy=ytop-i*(bh+gap)
        rect(ax,x0,yy,wmax,bh,fc="#ECECEC",ec="none",z=2)
        rect(ax,x0,yy,wmax*val,bh,fc=(CORE if hl else "#B9BEC4"),ec=INK,lw=0.6,z=3)
        ax.text(x0-0.9,yy+bh/2,lab,ha="right",va="center",fontsize=fs,
                color=INK if hl else DARK,weight="bold" if hl else "normal",zorder=4)
        ax.text(x0+wmax+0.7,yy+bh/2,f"{val:.2f}",ha="left",va="center",fontsize=fs,
                color=(CORE if hl else DARK),weight="bold" if hl else "normal",zorder=4)

def heat(ax,x0,ytop,cols,rows,M,cw=2.5,ch=4.0,fs=4.9):
    for j,c in enumerate(cols):
        ax.text(x0+j*cw+cw/2, ytop+1.2, c, ha="center",va="bottom",fontsize=fs,color=DARK)
    for i,rlab in enumerate(rows):
        yy=ytop-(i+1)*ch
        ax.text(x0-0.7, yy+ch/2, rlab, ha="right",va="center",fontsize=fs+0.4,color="#222")
        for j,v in enumerate(M[i]):
            rect(ax,x0+j*cw,yy,cw,ch,fc=GRID,ec="white",lw=1.0,z=3)
            ax.patches[-1].set_alpha(0.14+0.82*v)
            ax.text(x0+j*cw+cw/2,yy+ch/2,("%.2f"%v).lstrip("0") if v<1 else "1",
                    ha="center",va="center",fontsize=fs-0.3,
                    color="white" if v>0.5 else "#0b2a20",zorder=4)

def tree_glyph(ax,cx,cy,s=1.0,color=CORE):
    r=cy+3.2*s
    ax.plot([cx],[r],marker="o",ms=4.6*s,color=color,mec=INK,mew=0.8,zorder=6)
    for dx in (-3.4*s,3.4*s):
        ax.plot([cx,cx+dx],[r,cy+0.4*s],color=color,lw=1.3,zorder=5)
        ax.plot([cx+dx],[cy+0.4*s],marker="o",ms=3.7*s,color=color,mec=INK,mew=0.7,zorder=6)
        for dx2 in (-1.6*s,1.6*s):
            ax.plot([cx+dx,cx+dx+dx2],[cy+0.4*s,cy-2.6*s],color=color,lw=1.05,zorder=5)
            ax.plot([cx+dx+dx2],[cy-2.6*s],marker="s",ms=3.1*s,color=color,mec=INK,mew=0.6,zorder=6)


def wd_glyph(ax,cx,cy,s=1.0):
    seg=0.55*s; h=0.8*s; gap=0.32*s
    ax.add_patch(FancyBboxPatch((cx-0.7*s,cy-2*(h+gap)-0.5*s),5.4*s,3*h+2*gap+1.0*s,boxstyle="round,pad=0.15",fc="white",ec="#cfcfcf",lw=0.4,zorder=8))
    rows=[[(2,"#333"),(1,"#8c1b1b"),(2,"#333")],
          [(1,"#8c1b1b"),(2,"#333"),(1,"#8c1b1b"),(1,"#333")],
          [(2,"#333"),(1,"#333"),(2,"#8c1b1b")]]
    for r,row in enumerate(rows):
        y=cy-r*(h+gap); x=cx
        for w,c in row:
            ax.add_patch(Rectangle((x,y),w*seg,h,fc=c,ec="none",zorder=9)); x+=w*seg+gap


# ====================================================================
# 1) TASK figure -- the problem, grounded in the KG (multi-row example)
# =====================================================================
def fig_task():
    fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.set_xlim(0,100); ax.set_ylim(0,78)
    ax.set_aspect("equal"); ax.axis("off")

    headers=["Country","Capital","Population"]
    rows=[["France","Paris","68 M"],["Japan","Tokyo","125 M"],
          ["Egypt","Cairo","111 M"],["Brazil","Brasília","203 M"]]
    cw,ch=14.5,6.0; tx,hy=3,38
    ctr=table(ax,tx,hy,cw,ch,headers,rows,fs_h=6.8,fs_b=6.3); hdr_top=hy+ch
    ynode=60
    clschip(ax,ctr[0],ynode,"country\nQ6256",w=13,h=10,fs=6.3)
    clschip(ax,ctr[1],ynode,"capital city\nQ5119",w=13.4,h=10,fs=6.3)
    arrow(ax,ctr[0],ynode-5.2,ctr[0],hdr_top+0.3,c=CLS,lw=2.0,ms=12)
    arrow(ax,ctr[1],ynode-5.2,ctr[1],hdr_top+0.3,c=CLS,lw=2.0,ms=12)
    ax.text(ctr[2],ynode,"literal\nP1082",ha="center",va="center",fontsize=6.4,color=DARK,style="italic")
    ax.add_patch(FancyArrowPatch((ctr[0],ynode+5.5),(ctr[1],ynode+5.5),
                 connectionstyle="arc3,rad=-0.45",arrowstyle="-|>",mutation_scale=16,
                 color=PROP,lw=2.6,zorder=6))
    ax.text((ctr[0]+ctr[1])/2,ynode+15,"P36  capital",ha="center",va="center",
            fontsize=8.2,color=PROP_DK,weight="bold")

    gL=52
    rrect(ax,gL,16,46,50,fc="none",ec=MUTE,lw=1.1,r=1.0,ls=(0,(1,2.5)))
    ax.text(gL+23,62,"grounded in Wikidata",ha="center",fontsize=6.8,color=ENT,style="italic")
    ex1,ex2=gL+12,gL+34
    clschip(ax,ex1,53,"country",w=15,h=6.6,fs=6.2)
    clschip(ax,ex2,53,"capital city",w=16,h=6.6,fs=6.2)
    ell(ax,ex1,40,"France\nQ142",fc=ENT,w=16,h=8.6,fs=6.4)
    ell(ax,ex2,40,"Paris\nQ90",fc=ENT,w=15,h=8.6,fs=6.4)
    arrow(ax,ex1,44.4,ex1,49.7,c=CLS,lw=1.4,ms=9,ls=(0,(3,2)))
    arrow(ax,ex2,44.4,ex2,49.7,c=CLS,lw=1.4,ms=9,ls=(0,(3,2)))
    ax.text(ex1-1.2,47.2,"P31",ha="right",va="center",fontsize=5.4,color=CLS_DK)
    arrow(ax,ex1+8.4,40,ex2-7.8,40,c=PROP,lw=2.3,ms=12)
    ax.text((ex1+ex2)/2,36,"P36",ha="center",fontsize=6.0,color=PROP_DK,weight="bold")
    ax.text(gL+23,24,"a column's type (CTA) = the class its\ncell entities instantiate (P31 instance of)",
            ha="center",va="center",fontsize=6.0,color="#334")

    doth(ax,3,97,11,c=MUTE)
    rect(ax,6,4.6,3.6,3.6,fc=CLS_BG,ec=CLS,lw=1.7); ax.text(11.2,6.4,"column type (CTA)",fontsize=6.8,va="center",color=INK)
    arrow(ax,41,6.4,44.6,6.4,c=PROP,lw=2.4,ms=11); ax.text(46,6.4,"column relation (CPA)",fontsize=6.8,va="center",color=INK)
    ax.add_patch(Ellipse((80,6.4),4.6,3.8,fc=ENT,ec=INK,lw=1.2)); ax.text(83.2,6.4,"KG entity",fontsize=6.8,va="center",color=INK)
    fig.savefig(FIG/"task.pdf"); fig.savefig("/tmp/_task.png",dpi=200); plt.close(fig); print("wrote figures/task.pdf")


# =====================================================================
# 2) OVERVIEW figure -- minimalist TWO-ROW flow (dotted dividers, straight arrows)
# =====================================================================
def fig_overview():
    # Authored tall (ylim 0-84) with enlarged message fonts so it stays legible when the
    # figure* is scaled to text width. Dense heatmap cell values are supporting detail.
    fig,ax=plt.subplots(figsize=(9.8,8.2)); ax.set_xlim(0,100); ax.set_ylim(-6,84)
    ax.set_aspect("equal"); ax.axis("off")
    top=[17,50,83]
    for x in (33.5,66.5): dotv(ax,x,47,78)      # top-row dividers
    dotv(ax,50,-6,33)                           # bottom-row divider
    # straight flow arrows (top row)
    arrow(ax,27,61,39,61,c=INK,lw=2.8,ms=21); arrow(ax,63,61,71,61,c=INK,lw=2.8,ms=21)
    # short connector: Ranker (row 1, right) drops to Decode (row 2, right)
    ax.add_patch(FancyArrowPatch((90,42),(87,29.5),connectionstyle="arc3,rad=-0.4",
                 arrowstyle="-|>",mutation_scale=21,color=INK,lw=2.8,zorder=6,shrinkA=6,shrinkB=6))
    # bottom row flows right -> left: Decode -> Output
    arrow(ax,54,18,43,18,c=INK,lw=2.8,ms=21)

    # ---- 1 . Candidates (table + entity linking) ----
    steplabel(ax,17,80,1,"Candidates",SLATE)
    ct=table(ax,8,71,6.4,3.6,["Ctry","Cap","Pop"],
             [["France","Paris","68M"],["Japan","Tokyo","125M"]],fs_h=6.2,fs_b=5.8,hi=(0,1))
    arrow(ax,ct[1],63.8,ct[1],59.9,c=ENT,lw=2.0,ms=10,sh=1.5)
    ax.text(ct[1]+5.6,61.7,"link",ha="left",va="center",fontsize=6.0,color=ENT,style="italic")
    ell(ax,16,56,"Paris  Q90\ncapital city",fc=ENT,w=18,h=7.4,fs=6.4); check(ax,25.8,56,s=0.8,c=CLS)
    ax.text(27.1,56,"top-1",ha="left",va="center",fontsize=4.6,color=CLS_DK)
    wd_glyph(ax,2.6,60.9,s=0.72); ax.text(4.6,56.8,"Wikidata",ha="center",va="center",fontsize=4.5,color=ENT,style="italic")
    ell(ax,10.6,48.3,"Paris,TX\nQ830149",fc=MUTE,w=11,h=6.0,fs=5.2)
    ell(ax,23.4,48.3,"P.Hilton\nQ47899",fc=MUTE,w=11,h=6.0,fs=5.2)
    ax.text(17,41.6,"3 entity candidates (ambiguous)  ·  shared w/ SOTA",ha="center",fontsize=5.4,color=DARK,style="italic")

    # ---- 2 . closed-form feature matrix ----
    steplabel(ax,50,80,2,"Features",CORE)
    ax.text(50,76.2,"one closed-form vector per candidate type of the Ctry column",ha="center",fontsize=5.1,color=DARK,style="italic")
    cols=["cov","hier","lex","prior","freq"]; rlab=["country","state","capital","human"]
    M=[[.98,.90,.61,.40,.30],[.72,.40,.52,.18,.20],[.10,.05,.15,.05,.02],[.06,.00,.05,.02,.01]]
    heat(ax,45.6,73,cols,rlab,M,cw=3.2,ch=4.6,fs=6.6)
    rect(ax,45.3,68.2,16.2,4.8,fc="none",ec=CORE,lw=1.5,z=6)               # the winning 'country' row
    ax.text(62.2,70.6,"wins",ha="left",va="center",fontsize=4.6,color=CORE,weight="bold")
    ax.text(50,49.6,"read straight from Wikidata (no neural net):",ha="center",fontsize=5.4,color=DARK,style="italic")
    ax.text(50,46.4,"cov = type coverage · hier = hops climbed",ha="center",fontsize=4.5,color="#5b5b5b")
    ax.text(50,43.4,"lex = header-label cosine · prior/freq = training freq",ha="center",fontsize=4.5,color="#5b5b5b")

    # ---- 3 . boosted-tree ranker ----
    steplabel(ax,83,80,3,"Ranker",CORE)
    ax.text(83,75.4,"$\\phi$(country) = [ .98  .90  .61  .40  .30 ]",ha="center",fontsize=5.0,color=INK)
    ax.text(83,71.6,"sum of boosted trees",ha="center",fontsize=7.0,color=CORE,weight="bold")
    tree_glyph(ax,78.0,61.5,s=1.24,color=CORE); tree_glyph(ax,88.0,61.5,s=1.24,color=CORE)
    ax.text(83,53.3,"each split tests one feature $\\phi_i$",ha="center",fontsize=6.1,color=DARK,style="italic")
    ax.text(92.8,36.5,"score(country)\n= 0.94",ha="left",va="center",fontsize=5.0,color=CORE,weight="bold")
    # FLINT (light): feather icon + line, with a red 'no-GPU' circle-slash over GPU
    ax.add_patch(Ellipse((68.6,48.6),3.0,1.3,angle=35,fc=CLS_BG,ec=CLS,lw=0.8,zorder=5))
    ax.plot([67.6,69.7],[47.9,49.3],color=CLS,lw=0.6,zorder=6)
    ax.text(70.6,48.6,"tiny model  ·  CPU  ·",ha="left",va="center",fontsize=7.0,color=CORE,weight="bold")
    ax.text(89.4,48.6,"GPU",ha="left",va="center",fontsize=6.6,color="#8a8a8a",weight="bold")
    ax.add_patch(Ellipse((92.1,48.7),6.8,3.6,fc="none",ec="#c0392b",lw=1.3,zorder=8))
    ax.plot([88.9,95.3],[46.8,50.6],color="#c0392b",lw=1.3,zorder=8)
    # GRAMS+ (heavy): dumbbell icon + line
    ax.plot([67.8,69.4],[43.8,43.8],color="#7a7a7a",lw=2.4,zorder=5)
    ax.add_patch(Ellipse((67.5,43.8),1.6,1.6,fc="#7a7a7a",ec=INK,lw=0.5,zorder=6))
    ax.add_patch(Ellipse((69.7,43.8),1.6,1.6,fc="#7a7a7a",ec=INK,lw=0.5,zorder=6))
    ax.text(71.3,43.8,"vs. GRAMS+: GPU · 2 MLPs · Steiner-tree",ha="left",va="center",fontsize=5.6,color="#5b5b5b")

    # ---- 4 . decode (CTA argmax + CPA arborescence) -- RIGHT of bottom row ----
    steplabel(ax,70,35,4,"Decode",CORE)
    ax.text(70,30,"CTA: argmax type",ha="center",fontsize=6.9,color=INK,weight="bold")
    scorebars(ax,69.5,26.5,[("country",.94,True),("state",.72,False),("capital",.15,False)],
              wmax=8.8,bh=2.7,gap=3.1,fs=6.1)
    check(ax,83.6,27.8,s=1.05,c=CLS)
    ax.text(70,11.5,"CPA: max-weight arborescence",ha="center",fontsize=6.9,color=INK,weight="bold")
    gy=4.5; nx=[57,70,83]
    for x,lab in zip(nx,["Ctry","Cap","Pop"]):
        rect(ax,x-3.0,gy-2.3,6.0,4.6,fc="white",ec=INK,lw=1.2,z=4); ax.text(x,gy,lab,ha="center",va="center",fontsize=6.0,zorder=5)
    arrow(ax,nx[0]+3.3,gy,nx[1]-3.3,gy,c=CORE,lw=3.0,ms=11,sh=1.5)         # kept: Ctry -P36-> Cap
    ax.text((nx[0]+nx[1])/2,gy+3.6,"P36 .90",ha="center",fontsize=6.0,color=CORE,weight="bold")
    ax.text((nx[0]+nx[1])/2,gy+1.7,"kept",ha="center",fontsize=4.3,color=CORE)
    arrow(ax,nx[1]+3.3,gy,nx[2]-3.3,gy,c=MUTE,lw=1.7,ms=9,ls=(0,(2,2)),sh=1.5)  # Cap .. Pop (literal)
    ax.text((nx[1]+nx[2])/2,gy+2.9,"literal",ha="center",fontsize=5.4,color=DARK)
    # competing inverse edge (Cap -> Ctry), faint dashed, pruned by the arborescence
    ax.add_patch(FancyArrowPatch((nx[1]-3.3,gy-2.7),(nx[0]+3.3,gy-2.7),connectionstyle="arc3,rad=0.55",
                 arrowstyle="-|>",mutation_scale=8,color="#9AA0A6",lw=1.2,ls=(0,(2,2)),zorder=3,shrinkA=2,shrinkB=2))
    ax.text((nx[0]+nx[1])/2,gy-6.2,"P1376 .33  x  (pruned)",ha="center",fontsize=4.9,color="#8a8a8a")

    # ---- 5 . output semantic graph -- LEFT of bottom row ----
    steplabel(ax,30,33.5,5,"Output",CLS_DK)
    clschip(ax,30,25,"country\nQ6256",w=14,h=7.0,fs=6.6)
    ax.text(21.5,25,"Ctry\ncol.",ha="right",va="center",fontsize=4.6,color=SLATE,style="italic")
    clschip(ax,30,10,"capital city\nQ5119",w=14.8,h=7.0,fs=6.6)
    ax.text(21.3,10,"Cap\ncol.",ha="right",va="center",fontsize=4.6,color=SLATE,style="italic")
    arrow(ax,30,21.3,30,13.7,c=PROP,lw=2.6,ms=13)
    ax.text(31.6,17.5,"P36 capital",ha="left",fontsize=6.2,color=PROP_DK,weight="bold")
    ax.text(47,10,"Population\n= literal (P1082)",ha="center",va="center",fontsize=5.7,color=DARK)

    # ---- efficiency inset (cost, log) : FLINT vs GRAMS+ ----
    rrect(ax,41.5,20.5,17,12,fc="white",ec=MUTE,lw=1.0,r=0.8,ls=(0,(1,2.5)))
    ax.text(50,30.6,"cost (log)",ha="center",fontsize=4.9,color="#555",style="italic")
    rect(ax,43.5,26.9,2.4,1.9,fc=CORE,ec=INK,lw=0.5,z=5); ax.text(46.4,27.85,"FLINT",ha="left",va="center",fontsize=4.6,color=CORE,weight="bold",zorder=5)
    rect(ax,43.5,23.4,13.2,1.9,fc="#b9bec4",ec=INK,lw=0.5,z=5); ax.text(50.1,24.35,"GRAMS+",ha="left",va="center",fontsize=4.6,color="#555",zorder=5)
    ax.text(50,21.6,"~1000x lighter",ha="center",fontsize=5.0,color=INK,weight="bold")

    # NOTE: paper/flint/figures/overview.pdf is now the AI scientific-schematic combined figure
    # (from ai_schematic_concepts/flint_combined_specv1.png). This matplotlib version is written to
    # overview_vector.pdf so re-running does NOT clobber the AI overview.
    fig.savefig(FIG/"overview_vector.pdf"); fig.savefig("/tmp/_overview.png",dpi=200); plt.close(fig); print("wrote figures/overview_vector.pdf")



if __name__=="__main__":
    # NOTE: the paper's overview.pdf is the scientific-schematics (AI) figure, NOT py-generated.
    # Only the task figure is built here; fig_overview() is kept for reference but not called.
    fig_task(); print("done.")
