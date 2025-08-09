import ReactECharts from "echarts-for-react";

const KpiCard = ({ data }) => {
  const recent = data.slice(-12);
  const latest = recent.at(-1);

  const changeRates = recent.map((d, i) => {
    if (i === 0) return { 날짜: d.날짜, 상승률: 0 };
    const prev = recent[i - 1].값;
    const rate = ((d.값 - prev) / prev) * 100;
    return { 날짜: d.날짜, 상승률: parseFloat(rate.toFixed(2)) };
  });

  const option = {
    tooltip: {
      trigger: "axis",
      formatter: (params) => {
        const p = params[0];
        const idx = p.dataIndex;
        const change = changeRates[idx]?.상승률 ?? 0;
        return `
          ${p.axisValue}<br/>
          전월 대비 상승률: ${change.toFixed(2)}%
        `;
      },
    },
    grid: { top: 10, bottom: 25, left: 40, right: 10 },
    xAxis: {
      type: "category",
      data: changeRates.map((d) => d.날짜),
      axisLabel: { fontSize: 10 },
    },
    yAxis: {
      type: "value",
      axisLabel: { fontSize: 10, formatter: "{value}%" },
    },
    series: [
      {
        type: "line",
        data: changeRates.map((d) => [d.날짜, d.상승률]),
        lineStyle: {
          width: 2,
          color: "#3b82f6",
        },
        symbol: "circle",
        itemStyle: { color: "#3b82f6" },
        areaStyle: { opacity: 0.05, color: "#3b82f6" },
        smooth: true,
      },
    ],
  };

  return (
    <div
      style={{
        backgroundColor: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "8px",
        boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
        padding: "5px",
        marginTop: "2.5px",
        marginBottom: "2.5px",
        minHeight: "400px",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <h3
        style={{
          fontSize: "18px",
          fontWeight: "600",
          marginBottom: "16px",
          color: "#111827",
          textAlign: "center",
        }}
      >
        CPI 요약 지표
      </h3>

      <div style={{ textAlign: "center", marginBottom: "8px" }}>
        <div style={{ fontSize: "24px", fontWeight: "bold" }}>
          {latest?.값?.toFixed(1)}
        </div>
        <div style={{ fontSize: "13px", color: "#6b7280" }}>
          기준일: {latest?.날짜}
        </div>
      </div>

      <ReactECharts option={option} style={{ width: "100%", height: 300 }} />
    </div>
  );
};

export default KpiCard;
