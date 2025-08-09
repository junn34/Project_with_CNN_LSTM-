const KpiCard = ({ label, value, unit = "", color = "#111827" }) => (
  <div
    style={{
      flex: "1 1 45%",
      backgroundColor: "#f9fafb",
      border: "1px solid #e5e7eb",
      borderRadius: "8px",
      padding: "12px",
      boxSizing: "border-box",
      display: "flex",
      flexDirection: "column",
      gap: "6px"
    }}
  >
    <div style={{ fontSize: "13px", color: "#6b7280" }}>{label}</div>
    <div style={{ fontSize: "20px", fontWeight: "bold", color }}>
      {value !== undefined && value !== null ? value + unit : "N/A"}
    </div>
  </div>
);

const format = (val, digits = 2) =>
  typeof val === "number" ? val.toFixed(digits) : "N/A";

const KpiSummaryCards = ({
  news_count,
  sentiment_score,
  positive_ratio,
  negative_ratio
}) => {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "12px",
        justifyContent: "space-between",
        padding: "8px 4px",
        marginBottom: "8px"
      }}
    >
      <KpiCard label="총 뉴스 수" value={news_count} unit="건" />
      <KpiCard label="감성 점수" value={format(sentiment_score, 3)} />
      <KpiCard
        label="긍정 비율"
        value={format(positive_ratio * 100, 1)}
        unit="%"
        color="#3b82f6"
      />
      <KpiCard
        label="부정 비율"
        value={format(negative_ratio * 100, 1)}
        unit="%"
        color="#ef4444"
      />
    </div>
  );
};

export default KpiSummaryCards;
