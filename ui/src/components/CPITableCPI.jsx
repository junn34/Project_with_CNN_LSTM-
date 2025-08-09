import React from "react";
import cpiData from "../data/parsedCPIData_with_full_predictions.json";

const CPITableCPI = ({ startDate, endDate }) => {
  const records = cpiData["CPI"].filter(
    (r) => r.날짜 >= startDate && r.날짜 <= endDate
  );

  if (!records || records.length === 0) {
    return <div style={{ padding: 16 }}>해당 기간에 CPI 데이터가 없습니다.</div>;
  }

  const dates = records.map((r) => r.날짜);
  const actualRow = records.map((r) => r.값?.toFixed(2) ?? "-");
  const predictedRow = records.map((r) => r.예측값?.toFixed(2) ?? "-");
  const rateRow = records.map((r) => {
    if (typeof r.값 === "number" && typeof r.예측값 === "number" && r.값 !== 0) {
      const rate = ((r.예측값 - r.값) / r.값) * 100;
      return `${rate.toFixed(1)}%`;
    }
    return "-";
  });

  return (
    <div style={{ border: "1px solid #d1d5db", borderRadius: 8, padding: "1rem", backgroundColor: "#fff" }}>
      <h3 style={{ fontWeight: 700, fontSize: "16px", marginBottom: "12px" }}>CPI 상세 수치</h3>
      <div style={{ overflowX: "auto", width: "100%" }}>
        <table style={{
          tableLayout: "fixed",
          borderCollapse: "collapse",
          minWidth: `${dates.length * 140 + 150}px`,
          fontSize: "12px"
        }}>
          <thead>
            <tr style={{ backgroundColor: "#f1f5f9" }}>
              <th style={{ width: "150px", padding: "4px 6px", textAlign: "left", whiteSpace: "nowrap" }}>항목 / 날짜</th>
              {dates.map((d) => (
                <th key={d} style={{
                  width: "140px",
                  padding: "4px 6px",
                  textAlign: "center",
                  whiteSpace: "nowrap"
                }}>{d}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[["실제값", actualRow], ["예측값", predictedRow], ["예측률", rateRow]].map(([label, row]) => (
              <tr key={label}>
                <td style={{ padding: "4px 6px", fontWeight: 500, whiteSpace: "nowrap" }}>{label}</td>
                {row.map((v, i) => (
                  <td key={i} style={{
                    padding: "4px 6px",
                    textAlign: "center",
                    whiteSpace: "nowrap"
                  }}>{v}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CPITableCPI;
