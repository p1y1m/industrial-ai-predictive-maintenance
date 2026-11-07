export type Summary={total_assets:number;critical_assets:number;elevated_assets:number;anomalous_assets:number;median_predicted_rul:number;model_run_id:string;selected_model:string};
export type Asset={unit_id:number;predicted_rul:number;actual_rul:number;interval_lower:number;interval_upper:number;risk_15:number;risk_30:number;risk_60:number;anomaly_score:number;is_anomaly:boolean;health_state:'critical'|'elevated'|'nominal'};
export type ModelMetric={model:string;validation_rmse:number;test_rmse_capped:number;test_rmse_uncapped:number;test_mae_capped:number;nasa_score_uncapped:number;selected:boolean};
const base=import.meta.env.VITE_API_BASE_URL??'http://localhost:8000/api/v1';
const staticMode=import.meta.env.VITE_STATIC_DEMO==='true';
type DemoPayload={summary:Summary;fleet:{items:Asset[];total:number};models:ModelMetric[]};
let demoPromise:Promise<DemoPayload>|undefined;
const demo=()=>demoPromise??=fetch(`${import.meta.env.BASE_URL}data/demo.json`).then(r=>{if(!r.ok)throw new Error(`Offline data ${r.status}`);return r.json() as Promise<DemoPayload>});
async function get<T>(path:string):Promise<T>{const r=await fetch(`${base}${path}`);if(!r.ok)throw new Error(`API ${r.status}`);return r.json() as Promise<T>}
export const api={summary:()=>staticMode?demo().then(d=>d.summary):get<Summary>('/fleet/summary'),fleet:()=>staticMode?demo().then(d=>d.fleet):get<{items:Asset[];total:number}>('/fleet?page_size=100'),models:()=>staticMode?demo().then(d=>d.models):get<ModelMetric[]>('/models')};
