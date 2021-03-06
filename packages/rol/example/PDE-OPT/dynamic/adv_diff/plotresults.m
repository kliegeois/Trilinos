
adj = load('cell_to_node_quad.txt') + 1;  %% load node adjacency table, increment by 1 for 1-based indexing

nodes = load('nodes.txt');  %% load node coordinates

axsize = 400;

control = load('control.txt');
[nt,nc] = size(control);
Tmesh   = linspace(0,1,nt).';

figure('Position', [100 100 2*axsize 2*axsize]);
plot(Tmesh,control,'LineWidth',2)
axis square;
xlabel('t');
ylabel('z(t)');
set(gca, 'FontSize', 16); %set(gcf, 'Color', 'White');% tightfig;
legend({'x=0.25, y=0.25',...
        'x=0.50, y=0.25',...
        'x=0.75, y=0.25',...
        'x=0.25, y=0.50',...
        'x=0.50, y=0.50',...
        'x=0.75, y=0.50',...
        'x=0.25, y=0.75',...
        'x=0.50, y=0.75',...
        'x=0.75, y=0.75'}, 'Location', 'northeastoutside');
print('-depsc2','controls.eps');
%data = control(1)*exp(-0.5*( (nodes(:,1)-0.25).^2./(0.05)^2 + (nodes(:,2)-0.25).^2./(0.05)^2)) + ...
%       control(2)*exp(-0.5*( (nodes(:,1)-0.50).^2./(0.05)^2 + (nodes(:,2)-0.25).^2./(0.05)^2)) + ...
%       control(3)*exp(-0.5*( (nodes(:,1)-0.75).^2./(0.05)^2 + (nodes(:,2)-0.25).^2./(0.05)^2)) + ...
%       control(4)*exp(-0.5*( (nodes(:,1)-0.25).^2./(0.05)^2 + (nodes(:,2)-0.50).^2./(0.05)^2)) + ...
%       control(5)*exp(-0.5*( (nodes(:,1)-0.50).^2./(0.05)^2 + (nodes(:,2)-0.50).^2./(0.05)^2)) + ...
%       control(6)*exp(-0.5*( (nodes(:,1)-0.75).^2./(0.05)^2 + (nodes(:,2)-0.50).^2./(0.05)^2)) + ...
%       control(7)*exp(-0.5*( (nodes(:,1)-0.25).^2./(0.05)^2 + (nodes(:,2)-0.75).^2./(0.05)^2)) + ...
%       control(8)*exp(-0.5*( (nodes(:,1)-0.50).^2./(0.05)^2 + (nodes(:,2)-0.75).^2./(0.05)^2)) + ...
%       control(9)*exp(-0.5*( (nodes(:,1)-0.75).^2./(0.05)^2 + (nodes(:,2)-0.75).^2./(0.05)^2));
%trisurf(adj, nodes(:,1), nodes(:,2), data);
%shading interp;
%view(2);
%axis square;
%xlabel('x');
%ylabel('y');
%colorbar
%set(gca, 'FontSize', 16); set(gcf, 'Color', 'White');% tightfig;

figure('Position', [100 100 2*axsize 2*axsize]);
for i=0:nt-1
  data_state = importdata(['state.',int2str(i),'.txt'], ' ', 2);  %% we need to skip the first two lines
  trisurf(adj, nodes(:,1), nodes(:,2), data_state.data);
  shading interp;
  view(2);
  axis square;
  title(['Time t = ',num2str(i/(nt-1))],'fontsize',16);
  xlabel('x');
  ylabel('y');
  colorbar
  caxis([0,1.25])
  set(gca, 'FontSize', 16); set(gcf, 'Color', 'White');% tightfig;
  drawnow
end
