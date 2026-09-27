import type { Metadata } from "next";
import { DM_Sans, Space_Grotesk } from "next/font/google";
import "./globals.css";

// Space Grotesk carries display type: tight tracking, squared terminals.
const grotesk = Space_Grotesk({
  variable: "--font-grotesk",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

// DM Sans carries reading copy, which needs a rounder, quieter texture.
const dmSans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Diamond Price Prediction and Value Segmentation",
  description:
    "A complete data science pipeline on 53,772 diamond records: exploratory analysis, price regression, value-tier classification and clustering, with a live predictor and a step-by-step trace of how any price is produced.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${grotesk.variable} ${dmSans.variable} h-full`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
