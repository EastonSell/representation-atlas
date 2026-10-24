"""Build source-frame boundaries from measured final speech, never estimates."""
import math

def build_timeline(plan, timing, fps=30):
    if not isinstance(fps,int) or isinstance(fps,bool) or fps<=0:
        raise ValueError('fps must be a positive integer')
    planned=plan.get('segments',[]); spoken=timing.get('segments',[])
    ids=[s['id'] for s in planned]
    if not ids or len(ids)!=len(set(ids)) or ids!=[s['id'] for s in spoken]:
        raise ValueError('speech and plan must have identical, unique ordered IDs')
    result={**plan,'fps':fps,'segments':[]}; cursor=0
    for segment,audio in zip(planned,spoken):
        if segment['text']!=audio['text'] or segment.get('tts_text',segment['text'])!=audio.get('tts_text',audio['text']):
            raise ValueError('speech text is stale')
        duration=audio['duration']
        if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
            raise ValueError('speech duration must be positive and finite')
        frames=math.ceil(duration*fps)
        result['segments'].append({**segment,'start':cursor/fps,'end':(cursor+frames)/fps,'frames':frames})
        cursor+=frames
    result['duration']=cursor/fps
    return result
