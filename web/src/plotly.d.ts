declare module "plotly.js-dist-min" {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  type Any = any;
  const Plotly: {
    react: (el: HTMLElement, data: Any[], layout?: Any, config?: Any) => Promise<void>;
    purge: (el: HTMLElement) => void;
    Plots: { resize: (el: HTMLElement) => void };
  };
  export default Plotly;
}
