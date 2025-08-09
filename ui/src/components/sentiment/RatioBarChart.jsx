import ReactECharts from "echarts-for-react";

const RatioBarChart = ({ months, positive, negative, newsCount, N_pos, N_neg }) => {
  const option = {
    tooltip: {
      trigger: "axis",
      formatter: function (params) {
        const idx = params[0].dataIndex;
        return `
          <strong>${months[idx]}</strong><br/>
          총 뉴스 수: ${newsCount[idx]}건<br/>
          긍정 뉴스: ${N_pos[idx]}건 (${(positive[idx] * 100).toFixed(1)}%)<br/>
          부정 뉴스: ${N_neg[idx]}건 (${(negative[idx] * 100).toFixed(1)}%)
        `;
      }
    },
    legend: { top: 10, left: "center" },
    grid: {
      left: 40,
      right: 20,
      top: 60,
      bottom: 60  // ✅ 라벨 짤림 방지
    },
    xAxis: {
      type: "category",
      data: months,
      axisLabel: {
        rotate: 45,  // ✅ 겹침 방지
        margin: 12   // ✅ 라벨 간격 확보
      }
    },
    yAxis: {
      type: "value",
      max: 1,
      axisLabel: {
        formatter: (val) => `${(val * 100).toFixed(0)}%`
      }
    },
    series: [
      {
        name: "긍정",
        type: "bar",
        stack: "감성",
        data: positive,
        itemStyle: { color: "#3b82f6" }
      },
      {
        name: "부정",
        type: "bar",
        stack: "감성",
        data: negative,
        itemStyle: { color: "#ef4444" }
      }
    ]
  };

  return (
    <div
      style={{
        backgroundColor: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: "8px",
        padding: "12px",
        boxShadow: "0 2px 4px rgba(0,0,0,0.03)",
        height: "48%",
        minHeight: "260px"
      }}
    >
      <h3 style={{ textAlign: "center", fontSize: "16px", marginBottom: "8px" }}>
        긍/부정 비율 (월별)
      </h3>
      <ReactECharts option={option} style={{ width: "100%", height: "90%" }} />
    </div>
  );
};

export default RatioBarChart;
