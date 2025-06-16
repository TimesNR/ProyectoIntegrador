import React from 'react';
const BatchTable: React.FC = () => {
  return (
    <table className="table table-bordered">
    <thead className="table-light">
        <tr>
        <th>State</th>
        <th>April</th>
        <th>February</th>
        <th>January</th>
        <th>June</th>
        <th>March</th>
        <th>May</th>
        <th>Total</th>
        </tr>
    </thead>
    <tbody>
        <tr className="table-primary fw-bold">
        <td>California</td><td>$67,487</td><td>$60,118</td><td>$47,918</td><td>$52,293</td><td>$46,201</td><td>$59,061</td><td>$333,078</td>
        </tr>
        <tr>
        <td className="ps-4">San Francisco</td><td>$5,227</td><td>$5,179</td><td>$6,104</td><td>$6,221</td><td>$4,771</td><td>$5,158</td><td>$32,660</td>
        </tr>
        <tr>
        <td className="ps-4">Sacramento</td><td>$61,817</td><td>$54,611</td><td>$41,424</td><td>$45,647</td><td>$41,021</td><td>$53,434</td><td>$297,954</td>
        </tr>
        <tr>
        <td className="ps-4">Fresno</td><td>$443</td><td>$328</td><td>$390</td><td>$425</td><td>$409</td><td>$469</td><td>$2,464</td>
        </tr>

        <tr className="table-primary fw-bold">
        <td>Massachusetts</td><td>$69,718</td><td>$52,653</td><td>$52,411</td><td>$64,354</td><td>$48,528</td><td>$53,613</td><td>$341,277</td>
        </tr>
        <tr>
        <td className="ps-4">Boston</td><td>$7,392</td><td>$5,396</td><td>$7,693</td><td>$5,891</td><td>$5,994</td><td>$5,396</td><td>$37,312</td>
        </tr>
        <tr>
        <td className="ps-4">Salem</td><td>$62,326</td><td>$47,257</td><td>$44,718</td><td>$58,463</td><td>$42,534</td><td>$48,661</td><td>$303,959</td>
        </tr>

        <tr className="table-primary fw-bold">
        <td>Minnesota</td><td>$63,470</td><td>$55,634</td><td>$50,209</td><td>$51,598</td><td>$65,349</td><td>$66,509</td><td>$352,769</td>
        </tr>
        <tr>
        <td className="ps-4">Minneapolis</td><td>$6,622</td><td>$6,828</td><td>$6,231</td><td>$7,543</td><td>$4,942</td><td>$7,547</td><td>$39,713</td>
        </tr>
        <tr>
        <td className="ps-4">Rochester</td><td>$56,644</td><td>$48,556</td><td>$43,723</td><td>$43,734</td><td>$60,127</td><td>$58,572</td><td>$311,356</td>
        </tr>
        <tr>
        <td className="ps-4">Baxter</td><td>$204</td><td>$250</td><td>$255</td><td>$321</td><td>$280</td><td>$390</td><td>$1,700</td>
        </tr>
    </tbody>
    </table>

  );
};

export default BatchTable;